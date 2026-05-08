# HW3 실험 기록

## 1. 현재 Baseline

현재 기준 구현 파일:
[lora.h](/Users/hoyoung/Documents/yonsei/2026-1/멀티코어와GPU프로그래밍/mgp-2026-spring/hw3/experiment/lora.h)

현재 baseline은 `Step 1: thread당 output 2개 계산`, `Step 4: LoRA tail의 B_mat vectorized load`, `Step 3: kernel_xA reduction`을 도입한 버전이다. 채점 서버 환경에서 아래 조건으로 측정했다.

- GPU: NVIDIA GeForce RTX 3090
- 빌드 옵션: `nvcc -std=c++11 -arch=sm_86`
- 문제 크기: `B=32`, `in_dim=4096`, `out_dim=4096`, `r=8`

### 현재 baseline 5회 반복 측정 결과

| 회차 | GPU | 실행 시간(ms) | 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 1 | GPU 0 | 0.576 | 0.007324 |
| 2 | GPU 1 | 0.573 | 0.007324 |
| 3 | GPU 1 | 0.579 | 0.007324 |
| 4 | GPU 0 | 0.578 | 0.007324 |
| 5 | GPU 0 | 0.575 | 0.007324 |

정리:

- 최고 성능: `0.573 ms`
- 최악 성능: `0.579 ms`
- 평균 성능: `0.576 ms`
- 정확도: `PASS` (`0.007324 < 0.01`)

이 버전을 현재 제출 후보 baseline으로 사용한다.

## 2. 이전 Baseline과 비교

직전 baseline은 `OUTS_PER_THREAD=2`와 LoRA tail의 `B_mat float4 load`까지 적용했지만, `kernel_xA`는 기존의 1-thread-1-output 방식으로 계산하던 버전이었다.

### 직전 baseline 5회 반복 측정 결과

| 회차 | GPU | 실행 시간(ms) | 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 1 | GPU 0 | 0.679 | 0.005859 |
| 2 | GPU 0 | 0.684 | 0.005859 |
| 3 | GPU 1 | 0.681 | 0.005859 |
| 4 | GPU 1 | 0.674 | 0.005859 |
| 5 | GPU 0 | 0.676 | 0.005859 |

직전 baseline 정리:

- 최고 성능: `0.674 ms`
- 최악 성능: `0.684 ms`
- 평균 성능: `0.679 ms`
- 정확도: `PASS` (`0.005859 < 0.01`)

현재 baseline은 직전 baseline 대비 아래만큼 개선되었다.

- 최고 성능 기준: `0.674 ms -> 0.573 ms`
- 최악 성능 기준: `0.684 ms -> 0.579 ms`
- 평균 성능 기준: `0.679 ms -> 0.576 ms`
- 정확도: `0.005859 -> 0.007324`

채점은 5회 중 최대 실행 시간을 기준으로 하므로, 가장 중요한 비교값은 최악 성능이다. 현재 baseline은 직전 baseline 대비 worst-case 기준으로 약 `15.35%` 개선되었다.

이전 baseline은 `float4 load`, fused LoRA add, `xA` shared cache, `d_xA` lazy allocation까지 적용된 1-thread-1-output 버전이었다.

### 이전 baseline 5회 반복 측정 결과

| 회차 | GPU | 실행 시간(ms) | 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 1 | GPU 0 | 0.718 | 0.005859 |
| 2 | GPU 1 | 0.727 | 0.005859 |
| 3 | GPU 0 | 0.714 | 0.005859 |
| 4 | GPU 0 | 0.720 | 0.005859 |
| 5 | GPU 1 | 0.721 | 0.005859 |

이전 baseline 정리:

- 최고 성능: `0.714 ms`
- 최악 성능: `0.727 ms`
- 평균 성능: `0.720 ms`
- 정확도: `PASS` (`0.005859 < 0.01`)

현재 baseline은 이전 baseline 대비 아래만큼 개선되었다.

- 최고 성능 기준: `0.714 ms -> 0.573 ms`
- 최악 성능 기준: `0.727 ms -> 0.579 ms`
- 평균 성능 기준: `0.720 ms -> 0.576 ms`
- 정확도: `0.005859 -> 0.007324`

채점은 5회 중 최대 실행 시간을 기준으로 하므로, 가장 중요한 비교값은 최악 성능이다. 현재 baseline은 worst-case 기준으로 약 `20.36%` 개선되었다.

## 3. 현재 버전에서 도입한 것

현재 baseline은 WMMA/TF32 경로가 아니라, FP32 기반의 tiled shared-memory 커널 경로를 사용한다.

도입한 최적화는 아래와 같다.

1. 작은 커널로 먼저 `xA = x @ A.T`를 계산한다.
2. 메인 커널에서 `y = x @ W.T + scale * xA @ B.T`를 fused 형태로 계산한다.
3. `x`와 `W`를 K 방향으로 읽을 때 `float4` 벡터화 로드를 사용한다.
4. LoRA 항 계산 시 `xA`를 shared memory에 올려서 같은 block 내 반복 global load를 줄였다.
5. `d_xA`를 lazy allocation 하여, 측정 구간 안에서 반복적인 `cudaMalloc/cudaFree` 오버헤드를 제거했다.
6. `OUTS_PER_THREAD=2`를 도입하여 thread 하나가 output column 2개를 계산하도록 바꿨다.
7. LoRA tail에서 `B_mat`를 scalar 8회 load 대신 `float4` 2회 load로 읽도록 바꿨다.
8. `kernel_xA`를 block reduction 방식으로 바꿔 `xA[b, ri]` 하나를 32개 thread가 나눠 계산하도록 했다.

`Step 1`의 핵심은 `sX[ty][tk]`를 한 번 읽어 여러 output에 재사용하는 것이다. 이를 위해 accumulator를 `val[OUTS_PER_THREAD]`로 두고, `sW[tx + j * THREADS_X][tk]`를 함께 사용한다.

## 4. 왜 이 버전을 Baseline으로 잡는가

- 과제의 정확도 조건 `max abs error < 0.01`을 통과한다.
- 가장 큰 병목인 `x @ W.T`에 대해 tiled shared memory 최적화가 들어가 있다.
- LoRA 항을 마지막 store에 합쳐서 `y`에 대한 추가 round-trip이 없다.
- `float4` 로드로 K 루프에서 global load instruction 수를 줄였다.
- thread당 output 2개를 계산해 같은 `sX` 값을 더 많이 재사용한다.
- LoRA tail에서 `B_mat[o, 0..7]`를 `float4` 두 번으로 읽어 scalar load와 loop overhead를 줄였다.
- `xA = x @ A.T`에서 4096 길이 dot product를 여러 thread가 나눠 계산해 작은 커널의 latency를 크게 줄였다.
- 5회 반복 측정에서 편차가 작다.

채점은 5회 중 최대 실행 시간을 기준으로 하기 때문에, 최고 성능뿐 아니라 안정성도 중요하다.

## 5. 현재 Baseline 코드 스냅샷

현재 전체 구현은 아래와 같다. 이후 실험에서 성능이 나빠지거나 오차가 커지면 이 코드로 롤백한다.

```cpp
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
#define BLK_Y  8
#define OUTS_PER_THREAD 2
#define THREADS_X (BLK_X / OUTS_PER_THREAD)
#define TILE_K (BLK_X * 4)   // = 128; float4 loads, 32 K-iterations, 64 syncs
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
```

## 6. 직전 Baseline 코드 스냅샷

```cpp
#include "cuda_runtime.h"

// -- Tunable dimensions -------------------------------------------------------
// BLK_X  : output (N) tile width
// BLK_Y  : batch  (M) tile height
// OUTS_PER_THREAD : outputs computed by one thread along N
// THREADS_X : threadIdx.x width
// TILE_K : K-step per iteration = BLK_X * 4  (float4: 1 load covers 4 K values)
//
// sX load : BLK_Y x TILE_K elements, 2 float4s per thread  (128 threads x 8 = 1024)
// sW load : BLK_X x TILE_K elements, 8 float4s per thread
// -----------------------------------------------------------------------------
#define BLK_X  32
#define BLK_Y  8
#define OUTS_PER_THREAD 2
#define THREADS_X (BLK_X / OUTS_PER_THREAD)
#define TILE_K (BLK_X * 4)

static_assert(BLK_X % OUTS_PER_THREAD == 0,
              "BLK_X must be divisible by OUTS_PER_THREAD");
static_assert(THREADS_X * BLK_Y <= 1024, "threads/block exceeds CUDA limit");

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
        #pragma unroll
        for (int i = 0; i < 2; i++) {
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

        #pragma unroll
        for (int i = 0; i < 8; i++) {
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

    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    dim3 block(THREADS_X, BLK_Y);
    dim3 grid((out_dim + BLK_X - 1) / BLK_X, (B + BLK_Y - 1) / BLK_Y);
    kernel_xWT_fused<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);
}
```

## 7. 이전 Baseline 코드 스냅샷

이 코드는 이전 baseline이었던 1-thread-1-output 버전이다. 현재 기준보다 느리지만, 안정적인 fallback으로 남겨둔다.

```cpp
#include "cuda_runtime.h"

#define BLK_X  32
#define BLK_Y  8
#define TILE_K (BLK_X * 4)

static_assert(BLK_X % BLK_Y == 0,    "BLK_X must be divisible by BLK_Y");
static_assert(BLK_X * BLK_Y <= 1024, "threads/block exceeds CUDA limit");

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

    const int b = by * BLK_Y + ty;
    const int o = bx * BLK_X + tx;

    float val = 0.0f;

    if (tx < 8) {
        sXA[ty][tx] = (b < B) ? xA[b * r + tx] : 0.0f;
    }
    __syncthreads();

    for (int kt = 0; kt < K / TILE_K; kt++) {
        if (b < B) {
            float4 xv = *reinterpret_cast<const float4 *>(
                &x[b * K + kt * TILE_K + tx * 4]);
            sX[ty][tx * 4 + 0] = xv.x;
            sX[ty][tx * 4 + 1] = xv.y;
            sX[ty][tx * 4 + 2] = xv.z;
            sX[ty][tx * 4 + 3] = xv.w;
        } else {
            sX[ty][tx * 4 + 0] = 0.0f;
            sX[ty][tx * 4 + 1] = 0.0f;
            sX[ty][tx * 4 + 2] = 0.0f;
            sX[ty][tx * 4 + 3] = 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < BLK_X / BLK_Y; i++) {
            int w_local = ty + i * BLK_Y;
            int w_row   = bx * BLK_X + w_local;
            float4 wv = *reinterpret_cast<const float4 *>(
                &W[w_row * K + kt * TILE_K + tx * 4]);
            sW[w_local][tx * 4 + 0] = wv.x;
            sW[w_local][tx * 4 + 1] = wv.y;
            sW[w_local][tx * 4 + 2] = wv.z;
            sW[w_local][tx * 4 + 3] = wv.w;
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
            lora_val += sXA[ty][ri] * B_mat[o * r + ri];
        y[b * N + o] = val + scale * lora_val;
    }
}

void lora(float *d_x, float *d_W, float *d_A, float *d_B, float *d_y,
          int B, int in_dim, int out_dim, int r, float scale) {
    static float *d_xA = nullptr;
    if (d_xA == nullptr) {
        cudaMalloc(&d_xA, B * r * sizeof(float));
    }

    kernel_xA<<<B, r>>>(d_x, d_A, d_xA, B, in_dim, r);

    dim3 block(BLK_X, BLK_Y);
    dim3 grid((out_dim + BLK_X - 1) / BLK_X, (B + BLK_Y - 1) / BLK_Y);
    kernel_xWT_fused<<<grid, block>>>(d_x, d_W, d_xA, d_B, d_y,
                                      B, in_dim, out_dim, r, scale);
}
```

## 8. 앞으로의 개선 계획

### Step 0. 현재 버전 보존

목표:

- 정확도와 성능이 모두 검증된 fallback 버전을 유지한다.

실행 원칙:

- 현재 baseline을 덮어쓰기 전에 반드시 복사본이나 기록을 남긴다.

### Step 1. thread당 output 여러 개 계산

상태:

- 완료.
- `OUTS_PER_THREAD=2` 버전이 현재 baseline이다.

결과:

- worst-case가 `0.727 ms`에서 `0.688 ms`로 개선되었다.
- 정확도는 `0.005859`로 유지되었다.

### Step 2. block shape 튜닝

목표:

- 현재 문제 크기 `B=32`, `N=4096`에 더 잘 맞는 block shape를 찾는다.

계획:

- 아래 후보를 순서대로 비교한다.
  - `BLK_X=32, BLK_Y=8, OUTS_PER_THREAD=2`
  - `BLK_X=32, BLK_Y=4, OUTS_PER_THREAD=2`
  - `BLK_X=16, BLK_Y=8, OUTS_PER_THREAD=2`
  - `BLK_X=32, BLK_Y=8, OUTS_PER_THREAD=4`

기대 효과:

- occupancy, scheduling, data reuse 사이 균형이 더 좋아질 수 있다.

### Step 3. `kernel_xA` 개선

상태:

- 완료.
- `grid=(B, r)`, `block=32` 구조로 바꿔 `xA[b, ri]` 하나를 32개 thread가 나눠 계산한다.

결과:

- worst-case가 `0.684 ms`에서 `0.579 ms`로 개선되었다.
- 정확도는 `0.005859`에서 `0.007324`로 증가했지만, 과제 기준인 `0.01` 미만을 유지한다.

### `XA_THREADS` 튜닝 결과

| XA_THREADS | 최고 성능(ms) | 최악 성능(ms) | 평균 성능(ms) | 최대 절대 오차 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.569 | 0.603 | 0.582 | 0.007324 |
| 64 | 0.573 | 0.593 | 0.581 | 0.007202 |
| 32 | 0.573 | 0.579 | 0.576 | 0.007324 |

채점은 5회 중 최대 실행 시간을 기준으로 하므로, 최악 성능이 가장 낮은 `XA_THREADS=32`를 현재 baseline으로 선택한다.

목표:

- 작은 `xA` 커널의 비효율을 줄인다.

기존 계획:

- 현재 `<<<B, r>>> = <<<32, 8>>>` 스타일 대신 warp 친화적인 reduction 구조를 시도한다.
- `in_dim=4096` 축을 여러 thread가 나눠 처리하도록 바꾼다.

주의:

- 전체 시간에서 차지하는 비중은 작아서 우선순위는 메인 커널보다 낮다.

### Step 4. LoRA 쪽 `B_mat` 로드 최적화

상태:

- 완료.
- `B_mat[o * r : o * r + 8]`을 scalar 8회 load 대신 `float4` 2회 load로 읽는다.

결과:

- worst-case가 `0.688 ms`에서 `0.684 ms`로 개선되었다.
- 정확도는 `0.005859`로 유지되었다.

목표:

- fused tail 구간의 scalar memory traffic을 줄인다.

기존 계획:

- 정렬 조건이 안전하면 `B_mat[o * r : o * r + 8]`에 대해 vectorized load를 시도한다.
- `xA`는 현재처럼 shared memory 캐시를 유지한다.

### Step 5. unroll / register 튜닝

목표:

- K 루프 안쪽 연산을 조금 더 다듬는다.

계획:

- `#pragma unroll` 위치를 비교한다.
- 부분 수동 unroll을 실험한다.
- 성능이 나빠지면 register pressure 증가 여부를 의심한다.

### Step 6. 실험 전용 브랜치

후보:

- WMMA는 별도 실험 브랜치에서만 다시 본다.

현재 판단:

- 이번 과제에서는 FP32 tiled 경로가 메인 제출 경로로 더 적합하다.

이유:

- WMMA/TF32 경로는 정확도 조건을 만족시키기 어려웠고,
- 수동 TF32 변환을 넣으면 실행 시간이 커졌다.

## 9. 실험 시 체크 기준

향후 모든 실험은 아래 두 기준으로 현재 baseline과 비교한다.

- 성능 기준: 최악 실행 시간 `0.579 ms`보다 빨라야 한다.
- 정확도 기준: 최대 절대 오차 `0.01` 미만을 유지해야 한다.

추가로 기억할 점:

- 채점 서버는 5회 실행 중 최대 시간을 기준으로 사용한다.
- 따라서 최고 기록 1회보다, 반복 실행 시 흔들리지 않는 안정성이 더 중요하다.
