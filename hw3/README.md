# HW3 CUDA LoRA Optimization

## 구현 개요

`lora.h`는 LoRA 연산을 다음 두 단계로 나누어 처리한다.

```text
y = x @ W.T + scale * (x @ A.T) @ B.T
```

1. `kernel_xA`에서 작은 중간 행렬 `xA = x @ A.T`를 먼저 계산한다.
2. `kernel_xWT_fused`에서 `x @ W.T`와 `scale * xA @ B.T`를 하나의 커널 안에서 합쳐 최종 `y`를 저장한다.

문제 크기는 제공된 실행 코드 기준으로 `B=32`, `in_dim=4096`, `out_dim=4096`, `r=8`이다. 최종 설정은 다음과 같다.

```cpp
#define BLK_X 16
#define BLK_Y 16
#define OUTS_PER_THREAD 2
#define TILE_K 128
#define XA_THREADS 32
```

## 적용한 최적화

### 1. `xA = x @ A.T` 전용 reduction 커널

LoRA 항에서 필요한 `xA`는 크기가 `32 x 8`로 작지만, 각 원소는 길이 4096의 dot product이다. 따라서 `kernel_xA`는 `xA[b, ri]` 하나를 하나의 block이 맡고, 32개 thread가 `in_dim` 축을 나누어 계산한다.

각 thread가 부분합을 만든 뒤 shared memory reduction으로 합쳐서 `xA`에 저장한다. 기존의 1-thread-1-output 방식보다 dot product 병렬성이 좋아져 작은 커널의 실행 시간이 줄었다.

### 2. shared-memory tiled main kernel

가장 큰 연산인 `x @ W.T`는 `kernel_xWT_fused`에서 shared memory tile을 사용해 계산한다.

```cpp
__shared__ float sX[BLK_Y][TILE_K];
__shared__ float sW[BLK_X][TILE_K + 1];
```

`x`와 `W`를 `K` 방향으로 `TILE_K=128`씩 나누어 읽고, block 내부에서 재사용한다. `sW`에는 `TILE_K + 1` padding을 두어 shared memory bank conflict를 줄였다.

### 3. `float4` vectorized load

`x`, `W`, `B_mat`는 모두 연속적인 float 배열이다. 따라서 K 방향 load는 `float4`로 묶어 global memory load instruction 수를 줄였다.

`B_mat[o, 0..7]`도 scalar 8회 load 대신 `float4` 2회 load로 읽는다.

### 4. LoRA tail fusion

`x @ W.T`를 먼저 `y`에 저장하고 다시 LoRA 항을 더하는 방식은 `y`에 대한 global memory round-trip이 생긴다. 현재 구현은 main kernel 마지막 store 직전에 LoRA 값을 계산한다.

```text
y[b, o] = xWT_value + scale * dot(xA[b, :], B_mat[o, :])
```

이렇게 해서 별도의 LoRA add 커널과 `y` 재로드를 제거했다.

### 5. `xA` shared cache

main kernel에서 같은 block의 여러 output column이 같은 batch row의 `xA[b, 0..7]`를 사용한다. 따라서 block 시작 시 `xA`를 shared memory `sXA[BLK_Y][8]`에 올리고, LoRA tail에서 재사용한다.

### 6. block shape tuning

최종 block shape는 `BLK_X=16`, `BLK_Y=16`, `OUTS_PER_THREAD=2`이다. 이 설정은 block 하나가 `16 x 16 = 256`개 output을 계산하고, thread 수는 `THREADS_X * BLK_Y = 8 * 16 = 128`개이다.

기존 `BLK_X=32`, `BLK_Y=8`도 같은 256개 output을 계산하지만, `16 x 16`은 같은 `W` tile을 더 많은 batch row에서 재사용한다. 실험 결과 `32 x 8`보다 worst-case가 더 안정적으로 낮았다.

### 7. thread당 output 2개 계산

`OUTS_PER_THREAD=2`를 사용해 thread 하나가 같은 batch row에서 output column 2개를 누적한다. 이렇게 하면 같은 `sX[ty][tk]` 값을 두 output에 재사용할 수 있다.

### 8. `__launch_bounds__`

main kernel에는 다음 힌트를 추가했다.

```cpp
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void kernel_xWT_fused(...)
```

현재 `THREADS_PER_BLOCK=128`이므로, compiler에 block당 최대 thread 수와 SM당 최소 4 blocks 목표를 알려 register allocation과 occupancy 선택을 조정하도록 했다. 서버 실험에서 이 설정이 가장 좋은 성능을 보였다.

## 성능 측정

서버 환경:

```text
GPU: NVIDIA GeForce RTX 3090
Build: nvcc -std=c++11 -arch=sm_86
Problem size: B=32, in_dim=4096, out_dim=4096, r=8
```

10회 반복 측정 결과:

| 회차 | GPU | 실행 시간(ms) | 최대 절대 오차 |
| --- | --- | ---: | ---: |
| 1 | GPU 0 | 0.549 | 0.007324 |
| 2 | GPU 0 | 0.554 | 0.007324 |
| 3 | GPU 0 | 0.571 | 0.007324 |
| 4 | GPU 0 | 0.562 | 0.007324 |
| 5 | GPU 1 | 0.562 | 0.007324 |
| 6 | GPU 1 | 0.552 | 0.007324 |
| 7 | GPU 0 | 0.560 | 0.007324 |
| 8 | GPU 1 | 0.547 | 0.007324 |
| 9 | GPU 1 | 0.555 | 0.007324 |
| 10 | GPU 0 | 0.550 | 0.007324 |

정리:

| 항목 | 값 |
| --- | ---: |
| 최고 성능 | 0.547 ms |
| 최악 성능 | 0.571 ms |
| 평균 성능 | 0.556 ms |
| 최대 절대 오차 | 0.007324 |

정확도 기준인 `max abs error < 0.01`을 만족한다.
