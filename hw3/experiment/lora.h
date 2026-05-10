#include "cuda_runtime.h"

// Naive LoRA: one thread per output element (b, n).
// No shared memory, no vectorization, no kernel split.
// All data read directly from global memory every time.
#define BLOCK_SIZE 16

__global__ void kernel_lora_naive(
        const float *x,     // [B, K]
        const float *W,     // [N, K]
        const float *A,     // [r, K]
        const float *B_mat, // [N, r]
        float *y,           // [B, N]
        int B, int K, int N, int r, float scale)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= B || n >= N) return;

    // x @ W.T
    float out_linear = 0.0f;
    for (int k = 0; k < K; k++)
        out_linear += x[b * K + k] * W[n * K + k];

    // x @ A.T  (r=8, fixed-size stack array)
    float xA[8];
    for (int ri = 0; ri < r; ri++) {
        float v = 0.0f;
        for (int k = 0; k < K; k++)
            v += x[b * K + k] * A[ri * K + k];
        xA[ri] = v;
    }

    // xA @ B.T
    float out_lora = 0.0f;
    for (int ri = 0; ri < r; ri++)
        out_lora += xA[ri] * B_mat[n * r + ri];

    y[b * N + n] = out_linear + scale * out_lora;
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((out_dim + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (B      + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_lora_naive<<<grid, block>>>(d_x, d_W, d_A, d_B, d_y,
                                       B, in_dim, out_dim, r, scale);
}
