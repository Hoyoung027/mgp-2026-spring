#include "cuda_runtime.h"

// ── Tunable dimensions ────────────────────────────────────────────────────────
// BLK_X  : threadIdx.x → output (N) tile width
// BLK_Y  : threadIdx.y → batch  (M) tile height
// TILE_K : K-step per iteration = BLK_X * 4  (float4: 1 load covers 4 K values)
//
// sX load : BLK_Y × TILE_K elements, 1 float4 per thread   (256 threads × 4 = 1024 ✓)
// sW load : BLK_X × TILE_K elements, BLK_X/BLK_Y float4s per thread
//
// Constraints (static_assert enforces):
//   BLK_X % BLK_Y == 0        (integer sW load rounds)
//   BLK_X * BLK_Y <= 1024     (CUDA max threads/block)
//   K=4096 and N=4096 both divisible by TILE_K
// ─────────────────────────────────────────────────────────────────────────────
#define BLK_X  32
#define BLK_Y  8
#define TILE_K (BLK_X * 4)   // = 128; float4 loads, 32 K-iterations, 64 syncs

static_assert(BLK_X % BLK_Y == 0,       "BLK_X must be divisible by BLK_Y");
static_assert(BLK_X * BLK_Y <= 1024,    "threads/block exceeds CUDA limit");

// xA = x @ A.T
// x: [B, in_dim], A: [r, in_dim] → xA: [B, r]
// B=32, r=8 → 256 elements total; simple one-thread-per-output kernel
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

// y = x @ W.T + scale * xA @ B_mat.T  (fused, float4 tiled shared memory)
// x: [B, K], W: [N, K], xA: [B, r], B_mat: [N, r] → y: [B, N]
//
// TILE_K=128: each K iteration loads 128 values via float4
//   sX[BLK_Y][TILE_K]   : 1 float4/thread, 32 threads × 4 floats = 128 = one row ✓
//   sW[BLK_X][TILE_K+1] : (BLK_X/BLK_Y) float4s/thread, covers all 32 output rows ✓
//   +1 col padding: bank(sW[tx][tk]) = (tx*129+tk)%32, gcd(129,32)=1 → no conflict ✓
//
// K iterations: K/TILE_K = 32  (was 128),  __syncthreads: 64  (was 256)
// LoRA fused at final store: r=8 dot product, no extra y round-trip
__global__ void kernel_xWT_fused(const float *__restrict__ x,
                                  const float *__restrict__ W,
                                  const float *__restrict__ xA,
                                  const float *__restrict__ B_mat,
                                  float *y,
                                  int B, int K, int N, int r, float scale) {
    __shared__ float sX[BLK_Y][TILE_K];
    __shared__ float sW[BLK_X][TILE_K + 1];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b = by * BLK_Y + ty;
    const int o = bx * BLK_X + tx;

    float val = 0.0f;

    for (int kt = 0; kt < K / TILE_K; kt++) {
        // sX: 1 float4 per thread — 32 threads × 4 floats = 128 = one sX row
        // global load: x[b][kt*TILE_K + tx*4 .. tx*4+3], coalesced across warp
        if (b < B) {
            float4 xv = *reinterpret_cast<const float4 *>(&x[b * K + kt * TILE_K + tx * 4]);
            sX[ty][tx*4+0] = xv.x;  sX[ty][tx*4+1] = xv.y;
            sX[ty][tx*4+2] = xv.z;  sX[ty][tx*4+3] = xv.w;
        } else {
            sX[ty][tx*4+0] = sX[ty][tx*4+1] = sX[ty][tx*4+2] = sX[ty][tx*4+3] = 0.0f;
        }

        // sW: BLK_X/BLK_Y float4s per thread — covers all BLK_X output rows
        // round i: thread (ty,tx) loads sW[ty+i*BLK_Y][tx*4..tx*4+3]
        // global load: W[w_row][kt*TILE_K + tx*4 .. tx*4+3], coalesced across warp
        #pragma unroll
        for (int i = 0; i < BLK_X / BLK_Y; i++) {
            int w_local = ty + i * BLK_Y;
            int w_row   = bx * BLK_X + w_local;
            float4 wv = *reinterpret_cast<const float4 *>(&W[w_row * K + kt * TILE_K + tx * 4]);
            sW[w_local][tx*4+0] = wv.x;  sW[w_local][tx*4+1] = wv.y;
            sW[w_local][tx*4+2] = wv.z;  sW[w_local][tx*4+3] = wv.w;
        }

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; tk++)
            val += sX[ty][tk] * sW[tx][tk];

        __syncthreads();
    }

    if (b < B && o < N) {
        float lora_val = 0.0f;
        #pragma unroll
        for (int ri = 0; ri < 8; ri++)
            lora_val += xA[b * r + ri] * B_mat[o * r + ri];
        y[b * N + o] = val + scale * lora_val;
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    float *d_xA;
    cudaMalloc(&d_xA, B * r * sizeof(float));

    // Kernel 1: xA = x @ A.T  [32, 8]
    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    // Kernel 2: y = x @ W.T + scale * xA @ B.T (fused, tiled)
    dim3 block(BLK_X, BLK_Y);
    dim3 grid((out_dim + BLK_X - 1) / BLK_X, (B + BLK_Y - 1) / BLK_Y);
    kernel_xWT_fused<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);

    cudaFree(d_xA);
}