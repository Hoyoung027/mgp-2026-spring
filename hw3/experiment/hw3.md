# HW3 실험 기록

## 1. 현재 Baseline

현재 기준 구현 파일:
[lora.h](/Users/hoyoung/Documents/yonsei/2026-1/멀티코어와GPU프로그래밍/mgp-2026-spring/hw3/experiment/lora.h)

현재 baseline은 채점 서버 환경에서 아래 조건으로 측정했다.

- GPU: NVIDIA GeForce RTX 3090
- 빌드 옵션: `nvcc -std=c++11 -arch=sm_86`
- 문제 크기: `B=32`, `in_dim=4096`, `out_dim=4096`, `r=8`

### 5회 반복 측정 결과

| 회차 | GPU | 실행 시간(ms) | 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 1 | GPU 0 | 0.718 | 0.005859 |
| 2 | GPU 1 | 0.727 | 0.005859 |
| 3 | GPU 0 | 0.714 | 0.005859 |
| 4 | GPU 0 | 0.720 | 0.005859 |
| 5 | GPU 1 | 0.721 | 0.005859 |

정리:

- 최고 성능: `0.714 ms`
- 최악 성능: `0.727 ms`
- 평균 성능: `0.720 ms`
- 정확도: `PASS` (`0.005859 < 0.01`)

이 버전을 현재 제출 후보 baseline으로 사용한다.

## 2. 현재 버전에서 도입한 것

현재 baseline은 WMMA/TF32 경로가 아니라, FP32 기반의 tiled shared-memory 커널 경로를 사용한다.

도입한 최적화는 아래와 같다.

1. 작은 커널로 먼저 `xA = x @ A.T`를 계산한다.
2. 메인 커널에서 `y = x @ W.T + scale * xA @ B.T`를 fused 형태로 계산한다.
3. `x`와 `W`를 K 방향으로 읽을 때 `float4` 벡터화 로드를 사용한다.
4. LoRA 항 계산 시 `xA`를 shared memory에 올려서 같은 block 내 반복 global load를 줄였다.
5. `d_xA`를 lazy allocation 하여, 측정 구간 안에서 반복적인 `cudaMalloc/cudaFree` 오버헤드를 제거했다.

## 3. 왜 이 버전을 Baseline으로 잡는가

- 과제의 정확도 조건 `max abs error < 0.01`을 이미 통과한다.
- 가장 큰 병목인 `x @ W.T`에 대해 tiled shared memory 최적화가 들어가 있다.
- LoRA 항을 마지막 store에 합쳐서 `y`에 대한 추가 round-trip이 없다.
- `float4` 로드로 K 루프에서 global load instruction 수를 줄였다.
- 5회 반복 측정에서 편차가 작다.

채점은 5회 중 최대 실행 시간을 기준으로 하기 때문에, 최고 성능뿐 아니라 안정성도 중요하다.

## 4. 현재 코드 스냅샷

```cpp
#include "cuda_runtime.h"

// -- Tunable dimensions -------------------------------------------------------
// BLK_X  : threadIdx.x -> output (N) tile width
// BLK_Y  : threadIdx.y -> batch  (M) tile height
// TILE_K : K-step per iteration = BLK_X * 4  (float4: 1 load covers 4 K values)
//
// sX load : BLK_Y x TILE_K elements, 1 float4 per thread   (256 threads x 4 = 1024)
// sW load : BLK_X x TILE_K elements, BLK_X/BLK_Y float4s per thread
// -----------------------------------------------------------------------------
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

## 5. 앞으로의 개선 계획

### Step 0. 현재 버전 보존

목표:

- 정확도와 성능이 모두 검증된 fallback 버전을 유지한다.

이유:

- 이후 실험에서 성능이 좋아지지 않거나 오차가 커질 수 있다.
- 따라서 현재 버전은 항상 되돌아올 수 있는 기준점이어야 한다.

실행 원칙:

- 현재 baseline을 덮어쓰기 전에 반드시 복사본이나 기록을 남긴다.

### Step 1. thread당 output 여러 개 계산

목표:

- shared memory에서 읽은 `sX[ty][tk]`를 여러 output column에 재사용한다.
- thread당 연산량을 늘려 arithmetic intensity를 높인다.

계획:

- 메인 커널에서 thread 하나가 output 2개 또는 4개를 계산하도록 변경한다.
- 우선 실험 후보:
  - `blockDim = (16, 8)` + thread당 output 2개
  - `blockDim = (8, 8)` + thread당 output 4개

기대 효과:

- 현재의 1-thread-1-output 구조보다 연산 재사용성이 좋아질 수 있다.

주의:

- register 사용량이 늘어서 occupancy가 떨어질 수 있다.

### Step 2. block shape 튜닝

목표:

- 현재 문제 크기 `B=32`, `N=4096`에 더 잘 맞는 block shape를 찾는다.

계획:

- 아래 후보를 순서대로 비교한다.
  - `BLK_X=32, BLK_Y=8`
  - `BLK_X=16, BLK_Y=8`
  - `BLK_X=32, BLK_Y=4`
  - `BLK_X=16, BLK_Y=16`

기대 효과:

- occupancy, scheduling, data reuse 사이 균형이 더 좋아질 수 있다.

### Step 3. `kernel_xA` 개선

목표:

- 작은 `xA` 커널의 비효율을 줄인다.

계획:

- 현재 `<<<B, r>>> = <<<32, 8>>>` 스타일 대신 warp 친화적인 reduction 구조를 시도한다.
- `in_dim=4096` 축을 여러 thread가 나눠 처리하도록 바꾼다.

기대 효과:

- 고정 오버헤드를 조금 더 줄일 수 있다.

주의:

- 전체 시간에서 차지하는 비중은 작아서 우선순위는 메인 커널보다 낮다.

### Step 4. LoRA 쪽 `B_mat` 로드 최적화

목표:

- fused tail 구간의 scalar memory traffic을 줄인다.

계획:

- 정렬 조건이 안전하면 `B_mat[o * r : o * r + 8]`에 대해 vectorized load를 시도한다.
- `xA`는 현재처럼 shared memory 캐시를 유지한다.

기대 효과:

- 작은 폭이지만 fused LoRA 구간에서 추가 개선 가능성이 있다.

### Step 5. unroll / register 튜닝

목표:

- K 루프 안쪽 연산을 조금 더 다듬는다.

계획:

- `#pragma unroll` 위치를 비교한다.
- 부분 수동 unroll을 실험한다.
- 성능이 나빠지면 register pressure 증가 여부를 의심한다.

기대 효과:

- 컴파일러가 자동으로 잡지 못한 부분이 있으면 소폭 개선될 수 있다.

### Step 6. 실험 전용 브랜치

후보:

- WMMA는 별도 실험 브랜치에서만 다시 본다.

현재 판단:

- 이번 과제에서는 FP32 tiled 경로가 메인 제출 경로로 더 적합하다.

이유:

- WMMA/TF32 경로는 정확도 조건을 만족시키기 어려웠고,
- 수동 TF32 변환을 넣으면 실행 시간이 커졌다.

## 6. 실험 시 체크 기준

향후 모든 실험은 아래 두 기준으로 baseline과 비교한다.

- 성능 기준: 최악 실행 시간 `0.727 ms`보다 빨라야 한다.
- 정확도 기준: 최대 절대 오차 `0.01` 미만을 유지해야 한다.

추가로 기억할 점:

- 채점 서버는 5회 실행 중 최대 시간을 기준으로 사용한다.
- 따라서 최고 기록 1회보다, 반복 실행 시 흔들리지 않는 안정성이 더 중요하다.
