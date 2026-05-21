# HW4 CUDA Conv2d Experiment

## Implementation Summary

이번 HW4 구현은 과제의 고정 입력 조건에 맞추어 단순하지만 효과적인 병렬화를 사용했다.

- Direct Conv2d
  - 각 CUDA thread가 output pixel 하나를 계산한다.
  - 입력 조건이 `N=1`, `C=1`, `K=1`, `pad=0`, `stride=1`, `dilation=1`로 고정되어 있어 channel/output-channel에 대한 복잡한 병렬화는 사용하지 않았다.

- im2col
  - im2col output의 column 방향이 연속적인 점을 이용해 `float4` 기반 vectorized store를 사용했다.
  - thread 하나가 연속된 4개의 im2col 원소를 계산하고 한 번에 저장한다.

- GEMM
  - 실제 호출에서 `M=1`이므로 일반적인 2D GEMM 대신 one-row 전용 kernel을 사용했다.
  - output column 하나를 thread 하나가 담당하고, `K=9` 또는 `K=49` 길이의 dot product를 계산한다.

## Performance Results

측정 환경은 `NVIDIA GeForce RTX 3090`이며, 제출용 `hw4` 디렉터리에서 `make run`을 5회 반복했다. 시간 단위는 ms이다.

### Optimized Raw Results

| Run | GPU | 3x3 Direct | 3x3 im2col | 3x3 GEMM | 7x7 Direct | 7x7 im2col | 7x7 GEMM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GPU 0 | 0.216 | 0.436 | 0.239 | 0.218 | 1.955 | 0.977 |
| 2 | GPU 0 | 0.239 | 0.417 | 0.239 | 0.216 | 1.960 | 0.976 |
| 3 | GPU 0 | 0.222 | 0.437 | 0.238 | 0.217 | 1.953 | 0.976 |
| 4 | GPU 1 | 0.228 | 0.436 | 0.237 | 0.219 | 1.956 | 0.978 |
| 5 | GPU 0 | 0.223 | 0.420 | 0.241 | 0.217 | 1.955 | 0.976 |

### Optimized Summary

| Case | Avg Direct | Max Direct | Avg im2col | Avg GEMM | Avg im2col+GEMM | Max im2col+GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3x3 | 0.226 | 0.239 | 0.429 | 0.239 | 0.668 | 0.675 |
| 7x7 | 0.217 | 0.219 | 1.956 | 0.977 | 2.932 | 2.936 |

### Naive Baseline Raw Results

`hw4/experiment` 디렉터리에서 naive im2col과 naive 2D GEMM을 사용해 5회 반복 측정했다.

| Run | GPU | 3x3 Direct | 3x3 im2col | 3x3 GEMM | 7x7 Direct | 7x7 im2col | 7x7 GEMM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | GPU 1 | 0.217 | 0.537 | 0.303 | 0.216 | 2.485 | 1.011 |
| 2 | GPU 1 | 0.225 | 0.536 | 0.305 | 0.217 | 2.488 | 1.012 |
| 3 | GPU 0 | 0.230 | 0.538 | 0.305 | 0.218 | 2.487 | 1.011 |
| 4 | GPU 0 | 0.228 | 0.536 | 0.305 | 0.218 | 2.486 | 1.010 |
| 5 | GPU 0 | 0.226 | 0.534 | 0.304 | 0.216 | 2.486 | 1.012 |

### Naive Baseline Summary

| Case | Avg Direct | Max Direct | Avg im2col | Avg GEMM | Avg im2col+GEMM | Max im2col+GEMM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3x3 | 0.225 | 0.230 | 0.536 | 0.304 | 0.841 | 0.843 |
| 7x7 | 0.217 | 0.218 | 2.486 | 1.011 | 3.498 | 3.500 |

### Optimization Effect

| Case | im2col Avg | GEMM Avg | im2col+GEMM Avg |
| --- | ---: | ---: | ---: |
| 3x3 naive | 0.536 | 0.304 | 0.841 |
| 3x3 optimized | 0.429 | 0.239 | 0.668 |
| 7x7 naive | 2.486 | 1.011 | 3.498 |
| 7x7 optimized | 1.956 | 0.977 | 2.932 |

`float4` 기반 im2col은 평균 기준으로 `3x3`에서 약 20%, `7x7`에서 약 21%의 im2col 시간 감소를 보였다. `M=1` 특화 GEMM은 `3x3`에서 약 21% 개선되었고, `7x7`에서는 약 3% 개선되었다.

## Analysis

`float4`를 사용한 im2col과 `M=1` 특화 GEMM을 적용해 naive baseline보다 전체 im2col+GEMM 시간을 줄일 수 있었다. 그러나 최적화 후에도 전체 im2col+GEMM 방식은 direct Conv2d보다 느렸다. 특히 `7x7`에서는 im2col 단계만 평균 `1.956 ms`가 걸려 direct Conv2d 평균 `0.217 ms`보다 훨씬 컸다.

이 차이는 im2col 방식이 큰 중간 buffer를 global memory에 저장하고, GEMM 단계에서 다시 읽어야 하기 때문이다. `7x7`의 경우 im2col matrix는 `49 x 2042 x 2042`개의 float로 구성되며, 중간 데이터 크기만 약 800 MB 수준이다. 반면 direct Conv2d는 중간 buffer 없이 input을 읽고 바로 output을 계산한다.

따라서 일반적인 CNN처럼 `C`와 `K`가 큰 경우에는 im2col+GEMM이 유리할 수 있지만, 이번 과제처럼 `C=1`, `K=1`인 단일 채널 convolution에서는 direct Conv2d가 더 적합했다.
