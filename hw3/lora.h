#include "cuda_runtime.h"

#define BLK_X  16
#define BLK_Y  16
#define OUTS_PER_THREAD 2
#define THREADS_X (BLK_X / OUTS_PER_THREAD)
#define TILE_K 128           // float4 loads, 32 K-iterations, 64 syncs
#define THREADS_PER_BLOCK (THREADS_X * BLK_Y)
#define K_FLOAT4S (TILE_K / 4)
#define SX_FLOAT4S (BLK_Y * K_FLOAT4S)
#define SW_FLOAT4S (BLK_X * K_FLOAT4S)
#define SX_LOADS_PER_THREAD (SX_FLOAT4S / THREADS_PER_BLOCK)
#define SW_LOADS_PER_THREAD (SW_FLOAT4S / THREADS_PER_BLOCK)
#define XA_THREADS 32

static_assert(BLK_X % OUTS_PER_THREAD == 0,
              "BLK_X must be divisible by OUTS_PER_THREAD");
static_assert(THREADS_X * BLK_Y <= 1024, "threads/block exceeds CUDA limit");
static_assert(SX_FLOAT4S % THREADS_PER_BLOCK == 0,
              "sX float4 loads must divide evenly across block threads");
static_assert(SW_FLOAT4S % THREADS_PER_BLOCK == 0,
              "sW float4 loads must divide evenly across block threads");

// xA = x @ A.T
// One block computes one xA[b, ri], splitting the 4096-wide dot product.
__global__ void kernel_xA(const float *x, const float *A, float *xA,
                           int B, int in_dim, int r) {
    __shared__ float partial[XA_THREADS];

    const int b = blockIdx.x;
    const int ri = blockIdx.y;
    const int tid = threadIdx.x;

    float val = 0.0f;
    for (int k = tid; k < in_dim; k += XA_THREADS)
        val += x[b * in_dim + k] * A[ri * in_dim + k];

    partial[tid] = val;
    __syncthreads();

    #pragma unroll
    for (int stride = XA_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        xA[b * r + ri] = partial[0];
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
    const int o_base = bx * BLK_X + tx;

    float val[OUTS_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < OUTS_PER_THREAD; j++)
        val[j] = 0.0f;

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
            #pragma unroll
            for (int j = 0; j < OUTS_PER_THREAD; j++)
                val[j] += xv * sW[tx + j * THREADS_X][tk];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < OUTS_PER_THREAD; j++) {
        int o = o_base + j * THREADS_X;
        if (b < B && o < N) {
            const float4 b0 = *reinterpret_cast<const float4 *>(&B_mat[o * r]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&B_mat[o * r + 4]);
            float lora_val = sXA[ty][0] * b0.x + sXA[ty][1] * b0.y +
                             sXA[ty][2] * b0.z + sXA[ty][3] * b0.w +
                             sXA[ty][4] * b1.x + sXA[ty][5] * b1.y +
                             sXA[ty][6] * b1.z + sXA[ty][7] * b1.w;
            y[b * N + o] = val[j] + scale * lora_val;
        }
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    static float *d_xA = nullptr;
    if (d_xA == nullptr) {
        cudaMalloc(&d_xA, B * r * sizeof(float));
    }

    // Kernel 1: xA = x @ A.T  [32, 8]
    dim3 grid_xA(B, r);
    kernel_xA<<<grid_xA, XA_THREADS>>>(d_x, d_A, d_xA, B, in_dim, r);

    // Kernel 2: y = x @ W.T + scale * xA @ B.T (fused, tiled)
    dim3 block(THREADS_X, BLK_Y);
    dim3 grid((out_dim + BLK_X - 1) / BLK_X, (B + BLK_Y - 1) / BLK_Y);
    kernel_xWT_fused<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);
}
