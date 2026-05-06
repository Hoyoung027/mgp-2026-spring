#include "cuda_runtime.h"

#define TILE 32

// xA = x @ A.T
// x: [B, in_dim], A: [r, in_dim] → xA: [B, r]
// B=32, r=8 → only 256 output elements, simple kernel is fine
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

// y = x @ W.T  (tiled shared memory, stage-2 optimization)
// x: [B, K], W: [N, K] → y: [B, N]
// Thread (ty, tx) in block (bx, by) computes y[by*TILE+ty][bx*TILE+tx].
// sX[ty][tx] = x[b][kt*TILE+tx]      → coalesced (consecutive tx → consecutive k)
// sW[ty][tx] = W[bx*TILE+ty][kt*TILE+tx] → coalesced
// inner loop: val += sX[ty][tk] * sW[tx][tk]
// sW has +1 col padding to eliminate bank conflict on sW[tx][tk]
__global__ void kernel_xWT(const float *x, const float *W, float *y,
                            int B, int K, int N) {
    __shared__ float sX[TILE][TILE];
    __shared__ float sW[TILE][TILE + 1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int b = by * TILE + ty;
    int o = bx * TILE + tx;

    float val = 0.0f;

    for (int kt = 0; kt < K / TILE; kt++) {
        sX[ty][tx] = (b < B) ? x[b * K + kt * TILE + tx] : 0.0f;

        int w_row = bx * TILE + ty;
        sW[ty][tx] = (w_row < N) ? W[w_row * K + kt * TILE + tx] : 0.0f;

        __syncthreads();

        for (int tk = 0; tk < TILE; tk++)
            val += sX[ty][tk] * sW[tx][tk];

        __syncthreads();
    }

    if (b < B && o < N)
        y[b * N + o] = val;
}

// y += scale * xA @ B.T
// xA: [B, r], B_mat: [out_dim, r] → add to y: [B, out_dim]
// r=8 → inner loop is trivially short
__global__ void kernel_lora_add(const float *xA, const float *B_mat, float *y,
                                 int B, int out_dim, int r, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * out_dim) return;
    int b = idx / out_dim;
    int o = idx % out_dim;
    float val = 0.0f;
    for (int ri = 0; ri < r; ri++)
        val += xA[b * r + ri] * B_mat[o * r + ri];
    y[idx] += scale * val;
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    float *d_xA;
    cudaMalloc(&d_xA, B * r * sizeof(float));

    // Step 1: xA = x @ A.T  [B=32, r=8] — tiny, simple kernel
    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    // Step 2: y = x @ W.T  [32, 4096] — bottleneck, tiled kernel
    dim3 block(TILE, TILE);
    dim3 grid((out_dim + TILE - 1) / TILE, (B + TILE - 1) / TILE);
    kernel_xWT<<<grid, block>>>(d_x, d_W, d_y, B, in_dim, out_dim);

    // Step 3: y += scale * xA @ B.T  [32, 4096] — trivial (r=8)
    int total = B * out_dim;
    kernel_lora_add<<<(total + 255) / 256, 256>>>(d_xA, d_B, d_y, B, out_dim, r, scale);

    cudaFree(d_xA);
}
