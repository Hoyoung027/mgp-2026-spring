#include "cuda_runtime.h"

// ── Tunable block dimensions ──────────────────────────────────────────────────
// BLK_X : threadIdx.x → output (N) tile width, also K-step per iteration
// BLK_Y : threadIdx.y → batch (M) tile height
//
// sX load : BLK_Y × BLK_X elements, one load per thread         (always fits)
// sW load : BLK_X × BLK_X elements, BLK_X/BLK_Y loads per thread (generalised)
//
// Constraints (static_assert enforces):
//   BLK_X % BLK_Y == 0        (integer sW load rounds)
//   BLK_X * BLK_Y <= 1024     (CUDA max threads/block)
//   K=4096 and N=4096 both divisible by BLK_X
//
// Candidates for step 2: (32,32) (32,16) (32,8) (16,16)
// ─────────────────────────────────────────────────────────────────────────────
#define BLK_X 32
#define BLK_Y 8

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

// y = x @ W.T + scale * xA @ B_mat.T  (fused, tiled shared memory)
// x: [B, K], W: [N, K], xA: [B, r], B_mat: [N, r] → y: [B, N]
//
// sX[BLK_Y][BLK_X] : one load per thread, coalesced (tx → consecutive K)
// sW[BLK_X][BLK_X+1] : BLK_X/BLK_Y loads per thread, coalesced
//   +1 col padding eliminates bank conflict on sW[tx][tk] access
//
// LoRA fused at store (r=8 loop, #pragma unroll → 8 MADs, no extra y round-trip)
__global__ void kernel_xWT_fused(const float *__restrict__ x,
                                  const float *__restrict__ W,
                                  const float *__restrict__ xA,
                                  const float *__restrict__ B_mat,
                                  float *y,
                                  int B, int K, int N, int r, float scale) {
    __shared__ float sX[BLK_Y][BLK_X];
    __shared__ float sW[BLK_X][BLK_X + 1];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b = by * BLK_Y + ty;
    const int o = bx * BLK_X + tx;

    float val = 0.0f;

    for (int kt = 0; kt < K / BLK_X; kt++) {
        // sX: one load per thread
        sX[ty][tx] = (b < B) ? x[b * K + kt * BLK_X + tx] : 0.0f;

        // sW: BLK_X/BLK_Y loads per thread (rounds over output rows)
        #pragma unroll
        for (int i = 0; i < BLK_X / BLK_Y; i++) {
            int w_local = ty + i * BLK_Y;
            int w_row   = bx * BLK_X + w_local;
            sW[w_local][tx] = (w_row < N) ? W[w_row * K + kt * BLK_X + tx] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BLK_X; tk++)
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
