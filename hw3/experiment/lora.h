#include "cuda_runtime.h"
#include <mma.h>
using namespace nvcuda;

// ── WMMA TF32 tiled LoRA kernel ───────────────────────────────────────────────
// Tensor-core fragment shape: M=16, N=16, K=8
// Warp tile:   WARP_M × WARP_N = 2 × 2 WMMA fragments → 32×32 per block
// Block:       4 warps = 128 threads
// Shared mem:  sX[32][32] + sW[32][32] = 8KB  (well under 48KB L1 limit)
//
// Compute: y = x @ W.T + scale * xA @ B.T
//   Phase 1 (K-loop): accumulate c_frag = x_tile @ W_tile.T via mma_sync
//   Phase 2 (store):  write c_frag → sX, then fuse LoRA in global write
//
// W.T trick: load sW[n][k] = W[n][k] with col_major B fragment
//   → WMMA interprets column-major [K×N] which equals row-major [N×K] = W ✓
// ─────────────────────────────────────────────────────────────────────────────
#define WMMA_M        16
#define WMMA_N        16
#define WMMA_K        8
#define WARP_M        2
#define WARP_N        2
#define BLOCK_M       (WARP_M * WMMA_M)            // 32
#define BLOCK_N       (WARP_N * WMMA_N)            // 32
#define BLOCK_THREADS (WARP_M * WARP_N * 32)       // 128
#define CHUNK_K       32                            // K-step per outer iteration

__device__ __forceinline__ float to_tf32(float x) {
    return __float_to_tf32(x);
}

// xA = x @ A.T  [B, r]
__global__ void kernel_xA(const float *x, const float *A, float *xA,
                           int B, int in_dim, int r) {
    int b  = blockIdx.x;
    int ri = threadIdx.x;
    if (b >= B || ri >= r) return;
    float val = 0.0f;
    for (int k = 0; k < in_dim; k++)
        val += x[b * in_dim + k] * A[ri * in_dim + k];
    xA[b * r + ri] = val;
}

// y = x @ W.T + scale * xA @ B.T  (WMMA TF32 tiled, fused LoRA)
__global__ void kernel_wmma_lora(const float *__restrict__ x,
                                  const float *__restrict__ W,
                                  const float *__restrict__ xA,
                                  const float *__restrict__ B_mat,
                                  float *y,
                                  int B, int K, int N, int r, float scale) {
    // sX reused as output buffer after K-loop
    __shared__ float sX[BLOCK_M][CHUNK_K];   // [32][32] = 4KB
    __shared__ float sW[BLOCK_N][CHUNK_K];   // [32][32] = 4KB

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / WARP_N;
    const int warp_n  = warp_id % WARP_N;
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int chunk = 0; chunk < K / CHUNK_K; chunk++) {
        const int k0 = chunk * CHUNK_K;

        // Load sX: 128 threads × 8 elements = 1024 = 32×32 ✓
        #pragma unroll
        for (int i = 0; i < BLOCK_M * CHUNK_K / BLOCK_THREADS; i++) {
            int idx    = tid + i * BLOCK_THREADS;
            int m_loc  = idx / CHUNK_K;
            int k_loc  = idx % CHUNK_K;
            int m      = block_m + m_loc;
            sX[m_loc][k_loc] = (m < B) ? to_tf32(x[m * K + k0 + k_loc]) : 0.0f;
        }

        // Load sW: same pattern; sW[n][k] = W[n][k], interpreted col_major = W.T
        #pragma unroll
        for (int i = 0; i < BLOCK_N * CHUNK_K / BLOCK_THREADS; i++) {
            int idx    = tid + i * BLOCK_THREADS;
            int n_loc  = idx / CHUNK_K;
            int k_loc  = idx % CHUNK_K;
            int n      = block_n + n_loc;
            sW[n_loc][k_loc] = (n < N) ? to_tf32(W[n * K + k0 + k_loc]) : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int wk = 0; wk < CHUNK_K / WMMA_K; wk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::col_major> b_frag;

            // a: sX[warp_m*16 : (warp_m+1)*16][wk*8 : (wk+1)*8], row_major, stride=CHUNK_K
            wmma::load_matrix_sync(a_frag,
                                   &sX[warp_m * WMMA_M][wk * WMMA_K],
                                   CHUNK_K);
            // b: sW[warp_n*16 : (warp_n+1)*16][wk*8 : (wk+1)*8], col_major, stride=CHUNK_K
            //    col_major means WMMA reads sW as [K×N], which equals W[n][k] = W.T[k][n] ✓
            wmma::load_matrix_sync(b_frag,
                                   &sW[warp_n * WMMA_N][wk * WMMA_K],
                                   CHUNK_K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // Store c_frag back to sX (output: [BLOCK_M][BLOCK_N] = [32][32], stride=BLOCK_N)
    wmma::store_matrix_sync(&sX[warp_m * WMMA_M][warp_n * WMMA_N],
                             c_frag, BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    // Write to global y, fusing LoRA add
    #pragma unroll
    for (int i = 0; i < BLOCK_M * BLOCK_N / BLOCK_THREADS; i++) {
        int idx    = tid + i * BLOCK_THREADS;
        int m_loc  = idx / BLOCK_N;
        int n_loc  = idx % BLOCK_N;
        int m      = block_m + m_loc;
        int n      = block_n + n_loc;
        if (m < B && n < N) {
            float lv = 0.0f;
            #pragma unroll
            for (int ri = 0; ri < 8; ri++)
                lv += xA[m * r + ri] * B_mat[n * r + ri];
            y[m * N + n] = sX[m_loc][n_loc] + scale * lv;
        }
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    float *d_xA;
    cudaMalloc(&d_xA, B * r * sizeof(float));

    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    dim3 block(BLOCK_THREADS);
    dim3 grid((out_dim + BLOCK_N - 1) / BLOCK_N,
              (B      + BLOCK_M - 1) / BLOCK_M);
    kernel_wmma_lora<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);

    cudaFree(d_xA);
}
