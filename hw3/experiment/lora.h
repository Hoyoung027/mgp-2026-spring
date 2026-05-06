#include "cuda_runtime.h"

// ── Tunable dimensions ────────────────────────────────────────────────────────
// BLK_X  : output (N) tile width
// BLK_Y  : batch  (M) tile height
// OUTS_PER_THREAD : outputs computed by one thread along N
// THREADS_X : threadIdx.x width
// TILE_K : K-step per iteration = BLK_X * 4  (float4: 1 load covers 4 K values)
//
// sX load : BLK_Y x TILE_K elements, SX_LOADS_PER_THREAD float4s per thread
// sW load : BLK_X x TILE_K elements, SW_LOADS_PER_THREAD float4s per thread
//
// Constraints (static_assert enforces):
//   BLK_X % OUTS_PER_THREAD == 0
//   THREADS_X * BLK_Y <= 1024
//   K=4096 and N=4096 both divisible by TILE_K
// ─────────────────────────────────────────────────────────────────────────────
#define BLK_X  32
#define BLK_Y  4
#define OUTS_PER_THREAD 2
#define THREADS_X (BLK_X / OUTS_PER_THREAD)
#define TILE_K (BLK_X * 4)   // = 128; float4 loads, 32 K-iterations, 64 syncs
#define THREADS_PER_BLOCK (THREADS_X * BLK_Y)
#define K_FLOAT4S (TILE_K / 4)
#define SX_FLOAT4S (BLK_Y * K_FLOAT4S)
#define SW_FLOAT4S (BLK_X * K_FLOAT4S)
#define SX_LOADS_PER_THREAD (SX_FLOAT4S / THREADS_PER_BLOCK)
#define SW_LOADS_PER_THREAD (SW_FLOAT4S / THREADS_PER_BLOCK)

static_assert(BLK_X % OUTS_PER_THREAD == 0,
              "BLK_X must be divisible by OUTS_PER_THREAD");
static_assert(THREADS_X * BLK_Y <= 1024, "threads/block exceeds CUDA limit");
static_assert(SX_FLOAT4S % THREADS_PER_BLOCK == 0,
              "sX float4 loads must divide evenly across block threads");
static_assert(SW_FLOAT4S % THREADS_PER_BLOCK == 0,
              "sW float4 loads must divide evenly across block threads");

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
//   sX[BLK_Y][TILE_K]   : loaded with SX_LOADS_PER_THREAD float4s/thread
//   sW[BLK_X][TILE_K+1] : loaded with SW_LOADS_PER_THREAD float4s/thread
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
    __shared__ float sXA[BLK_Y][8];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    const int b = by * BLK_Y + ty;
    const int o0 = bx * BLK_X + tx;
    const int o1 = o0 + THREADS_X;

    float val0 = 0.0f;
    float val1 = 0.0f;

    if (tx < 8) {
        sXA[ty][tx] = (b < B) ? xA[b * r + tx] : 0.0f;
    }
    __syncthreads();

    for (int kt = 0; kt < K / TILE_K; kt++) {
        // sX: fill the whole [BLK_Y][TILE_K] tile with linearized float4 loads.
        #pragma unroll
        for (int i = 0; i < SX_LOADS_PER_THREAD; i++) {
            int idx4 = tid + i * (THREADS_X * BLK_Y);
            int x_row_local = idx4 / (TILE_K / 4);
            int k4 = idx4 % (TILE_K / 4);
            int x_row = by * BLK_Y + x_row_local;

            if (x_row < B) {
                float4 xv = *reinterpret_cast<const float4 *>(
                    &x[x_row * K + kt * TILE_K + k4 * 4]);
                sX[x_row_local][k4 * 4 + 0] = xv.x;
                sX[x_row_local][k4 * 4 + 1] = xv.y;
                sX[x_row_local][k4 * 4 + 2] = xv.z;
                sX[x_row_local][k4 * 4 + 3] = xv.w;
            } else {
                sX[x_row_local][k4 * 4 + 0] = 0.0f;
                sX[x_row_local][k4 * 4 + 1] = 0.0f;
                sX[x_row_local][k4 * 4 + 2] = 0.0f;
                sX[x_row_local][k4 * 4 + 3] = 0.0f;
            }
        }

        // sW: fill the whole [BLK_X][TILE_K] tile with linearized float4 loads.
        #pragma unroll
        for (int i = 0; i < SW_LOADS_PER_THREAD; i++) {
            int idx4 = tid + i * (THREADS_X * BLK_Y);
            int w_local = idx4 / (TILE_K / 4);
            int k4 = idx4 % (TILE_K / 4);
            int w_row = bx * BLK_X + w_local;

            if (w_row < N) {
                float4 wv = *reinterpret_cast<const float4 *>(
                    &W[w_row * K + kt * TILE_K + k4 * 4]);
                sW[w_local][k4 * 4 + 0] = wv.x;
                sW[w_local][k4 * 4 + 1] = wv.y;
                sW[w_local][k4 * 4 + 2] = wv.z;
                sW[w_local][k4 * 4 + 3] = wv.w;
            } else {
                sW[w_local][k4 * 4 + 0] = 0.0f;
                sW[w_local][k4 * 4 + 1] = 0.0f;
                sW[w_local][k4 * 4 + 2] = 0.0f;
                sW[w_local][k4 * 4 + 3] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; tk++) {
            float xv = sX[ty][tk];
            val0 += xv * sW[tx][tk];
            val1 += xv * sW[tx + THREADS_X][tk];
        }

        __syncthreads();
    }

    if (b < B && o0 < N) {
        float lora_val0 = 0.0f;
        #pragma unroll
        for (int ri = 0; ri < 8; ri++)
            lora_val0 += sXA[ty][ri] * B_mat[o0 * r + ri];
        y[b * N + o0] = val0 + scale * lora_val0;
    }

    if (b < B && o1 < N) {
        float lora_val1 = 0.0f;
        #pragma unroll
        for (int ri = 0; ri < 8; ri++)
            lora_val1 += sXA[ty][ri] * B_mat[o1 * r + ri];
        y[b * N + o1] = val1 + scale * lora_val1;
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    static float *d_xA = nullptr;
    if (d_xA == nullptr) {
        cudaMalloc(&d_xA, B * r * sizeof(float));
    }

    // Kernel 1: xA = x @ A.T  [32, 8]
    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    // Kernel 2: y = x @ W.T + scale * xA @ B.T (fused, tiled)
    dim3 block(THREADS_X, BLK_Y);
    dim3 grid((out_dim + BLK_X - 1) / BLK_X, (B + BLK_Y - 1) / BLK_Y);
    kernel_xWT_fused<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);
}
