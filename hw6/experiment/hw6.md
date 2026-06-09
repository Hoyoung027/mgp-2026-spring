# HW6 Triton ResNet18 Report

## 1. 과제 목표 및 구현 개요

이번 과제의 목표는 PyTorch로 작성된 ResNet18의 주요 layer 연산을 직접 작성한 Triton kernel로 대체하는 것이다. PyTorch는 내부적으로 `cuDNN`, `cuBLAS`, `CUTLASS`와 같은 고성능 CUDA library를 활용하지만, 이번 과제에서는 이러한 library kernel에 의존하지 않고 Triton을 이용해 GPU kernel을 직접 구현하였다.

구현 대상 layer는 다음과 같다.

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d`
- `AvgPool2d`
- `Linear`

과제의 성능 기준은 ResNet18 모델이 CIFAR10 test data에서 최소 `75%` 이상의 accuracy를 달성하고, inference time이 `200 ms` 이내에 들어오는 것이다.

## 2. 구현 방식

### Conv2d

`Conv2d`는 direct convolution 방식으로 구현하였다. Output tensor의 spatial position과 output channel을 tile 단위로 나누고, 각 tile에서 input channel과 kernel 영역에 해당하는 `C * R * S` 축을 따라 값을 누적한다. 누적 연산은 Triton의 `tl.dot`을 사용해 matrix multiplication과 유사한 형태로 수행하였다.

구현에서는 하나의 generic Conv2d kernel을 사용하였다. ResNet18에서는 `7x7`, `3x3`, `1x1` convolution이 모두 사용되지만, 최종 제출 후보에서는 별도의 `1x1` fast path를 두지 않고 동일한 generic kernel로 처리하였다. Tile parameter는 다음과 같이 설정하였다.

```text
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32
```

Kernel launch grid는 output spatial position 전체와 output channel을 기준으로 구성하였다. `M = N * P * Q`로 두고, 하나의 Triton program이 `BLOCK_M`개의 output 위치와 `BLOCK_N`개의 output channel을 계산한다. `tl.program_id(0)`은 output 위치 tile을, `tl.program_id(1)`은 output channel tile을 담당한다.

각 output index는 다음과 같이 다시 tensor 좌표로 변환된다.

```text
q = offs_m % Q
p = (offs_m // Q) % P
n = offs_m // (P * Q)
```

그 후 `C * R * S` 축을 `BLOCK_K` 단위로 순회한다. 각 `cks` 값은 input channel `c`, kernel row `r`, kernel column `s`로 변환된다.

```python
s = cks % S
r = (cks // S) % R
c = cks // (R * S)
```

이를 이용해 실제 input coordinate인 `ih`, `iw`를 계산한다. Padding과 dilation을 고려하기 위해 다음 식을 사용하였다.

```python
ih = p * stride_h + r * dilation_h - pad_h
iw = q * stride_w + s * dilation_w - pad_w
```

Padding 영역이나 output tile의 boundary를 넘어가는 접근은 `mask`로 걸러내고, invalid element는 `0.0`으로 load한다. Weight도 output channel과 `C * R * S` 범위를 기준으로 mask를 적용한다. 이렇게 load한 input tile과 weight tile을 `tl.dot`으로 곱해 `acc`에 누적하고, 마지막에 output tensor의 `(n, k, p, q)` 위치에 저장한다.

핵심 연산 부분은 다음과 같다.

```python
input_offsets = ((n[:, None] * C + c[None, :]) * H + ih) * W + iw
x = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

weight_offsets = ((offs_out_k[None, :] * C + c[:, None]) * R + r[:, None]) * S + s[:, None]
w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

acc += tl.dot(x, w)
```

Output tensor는 `mgp.zeros`로 생성하였다. Bias는 과제의 ResNet18 `Conv2d` layer들이 `bias=False`로 구성되어 있으므로 별도로 처리하지 않았다.

### BatchNorm2d

`BatchNorm2d`는 inference mode 기준으로 구현하였다. 즉 batch마다 mean과 variance를 새로 계산하지 않고, pretrained model에 저장되어 있는 `running_mean`, `running_var`, `weight`, `bias`를 사용한다. 각 element에 대해 다음 형태의 normalization을 수행한다.

```text
y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
```

BatchNorm kernel은 input tensor 전체를 1차원으로 flatten한 관점에서 처리한다. 하나의 program은 `BLOCK_SIZE = 1024`개의 element를 담당하고, `offsets`를 통해 각 element의 linear index를 계산한다. NCHW layout에서 channel index는 다음 식으로 복원된다.

```python
channel = (offsets // (H * W)) % C
```

이 channel index를 사용해 `weight`, `bias`, `running_mean`, `running_var`를 load한다. 연산은 numerical stability를 위해 `tl.float32`로 수행하고, `tl.rsqrt(var + eps)`를 이용해 inverse square root를 계산한다. Output tensor는 input과 같은 shape로 `mgp.empty`를 통해 할당한다.

Batch normalization의 실제 계산은 다음과 같이 구현하였다.

```python
x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
gamma = tl.load(weight_ptr + channel, mask=mask, other=1.0).to(tl.float32)
beta = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
var = tl.load(var_ptr + channel, mask=mask, other=1.0).to(tl.float32)

y = (x - mean) * tl.rsqrt(var + eps) * gamma + beta
tl.store(out_ptr + offsets, y, mask=mask)
```

### ReLU, Pooling, Linear

`ReLU`는 input tensor의 각 element에 대해 `max(x, 0)`을 적용하는 elementwise Triton kernel로 구현하였다.

ReLU kernel도 tensor를 1차원 배열로 보고 처리한다. 하나의 program은 `BLOCK_SIZE = 1024`개의 element를 load하고, `tl.where(a > 0, a, 0)`으로 음수 값을 0으로 바꾼 뒤 store한다. `inplace=True`인 경우 output pointer를 input과 동일하게 사용하고, 그렇지 않은 경우 `mgp.empty`로 새 output tensor를 만든다.

```python
a = tl.load(in_ptr + offsets, mask=offsets < num_elements, other=0.0)
a = tl.where(a > 0, a, 0)
tl.store(out_ptr + offsets, a, mask=offsets < num_elements)
```

`MaxPool2d`와 `AvgPool2d`는 output 위치별 pooling window를 순회하면서 각각 maximum과 average를 계산하는 reduction kernel로 구현하였다.

Pooling kernel은 output tensor의 각 element를 기준으로 `n`, `c`, `p`, `q` 좌표를 복원한다. 이후 pooling window의 `kh`, `kw`를 순회하면서 input coordinate를 계산한다. Padding 때문에 input 범위를 벗어날 수 있으므로 `valid` mask를 사용한다. `MaxPool2d`는 accumulator를 `-inf`로 초기화하고 `tl.maximum`으로 갱신한다. `AvgPool2d`는 accumulator를 `0`으로 초기화한 뒤 valid input 값을 더하고, 마지막에 `KH * KW`로 나눈다. ResNet18의 마지막 average pooling은 `7x7` window를 사용해 `[N, 512, 7, 7]`을 `[N, 512, 1, 1]`로 줄인다.

Pooling의 window 순회 구조는 다음과 같다.

```python
for kh in range(0, KH):
    ih = p * SH + kh - PH
    h_mask = (ih >= 0) & (ih < H)
    for kw in range(0, KW):
        iw = q * SW + kw - PW
        valid = mask & h_mask & (iw >= 0) & (iw < W)
        x_offsets = ((n * C + c) * H + ih) * W + iw
```

`Linear`는 input matrix와 weight matrix의 multiplication에 bias를 더하는 형태로 구현하였다. Triton kernel 내부에서는 output matrix를 tile 단위로 나누고, `tl.dot`을 이용해 matrix multiplication을 수행하였다.

Linear kernel은 output matrix의 row tile과 column tile을 각각 `tl.program_id(0)`, `tl.program_id(1)`에 매핑한다. `BLOCK_M = 16`, `BLOCK_N = 16`, `BLOCK_K = 32`를 사용하였고, K dimension을 `BLOCK_K` 단위로 순회하면서 input tile과 weight tile을 load한다. 각 tile multiplication 결과는 `tl.float32` accumulator에 누적된다. K loop가 끝난 뒤 bias를 output column 기준으로 load해 더하고, boundary mask를 적용하여 output matrix에 저장한다.

```python
for k0 in range(0, K, BLOCK_K):
    k = k0 + offs_k
    a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :], mask=..., other=0.0)
    b = tl.load(b_ptr + k[:, None] * N + offs_n[None, :], mask=..., other=0.0)
    acc += tl.dot(a, b)

bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
acc += bias[None, :]
```

## 3. 실험 환경 및 결과

실험은 다음 환경에서 수행하였다.

- Server: `server8`
- Working directory: `~/mgp-2026-spring/hw6/experiment`
- Command: `make triton`
- Model: Triton ResNet18
- Accuracy target: `75%`
- Time target: `200 ms`

### PyTorch Baseline

먼저 PyTorch로 작성된 reference model을 실행한 결과는 다음과 같다.

| Model | Inference Time (ms) | Accuracy (%) |
|---|---:|---:|
| PyTorch ResNet18 | 28.6428 | 79.73 |

PyTorch baseline은 `28.6428 ms`의 inference time과 `79.73%`의 accuracy를 보였다.

### Triton 제출 후보: `BLOCK_K = 32`

최종 제출 후보로 선택한 Triton Conv2d 설정은 다음과 같다.

- Single generic Conv2d kernel
- No separate `1x1` Conv2d fast path
- `BLOCK_M = 32`
- `BLOCK_N = 32`
- `BLOCK_K = 32`

Triton cache를 삭제한 뒤 첫 실행을 수행했기 때문에, 첫 번째 run에는 Triton JIT compilation 및 cache generation 시간이 포함된다. 이후 run은 compile된 kernel이 cache되어 있는 상태에서 수행되었다.

| Run | Inference Time (ms) | Accuracy (%) | Note |
|---:|---:|---:|---|
| 1 | 180.7786 | 79.83 | Cold run, includes JIT/cache generation |
| 2 | 43.6268 | 79.83 | Cached run |
| 3 | 43.4623 | 79.83 | Cached run |
| 4 | 43.5569 | 79.83 | Cached run |
| 5 | 43.4237 | 79.83 | Cached run |
| 6 | 43.4997 | 79.83 | Cached run |

`BLOCK_K = 32` 설정에서는 cold run에서도 `180.7786 ms`가 측정되어 `200 ms` 기준을 만족하였다. Cached run에서는 약 `43.4-43.6 ms`로 안정적인 inference time을 보였다. Accuracy는 모든 run에서 `79.83%`로 측정되어 정확도 기준도 만족하였다.

### 비교 실험: `BLOCK_K = 64`

추가로 같은 generic Conv2d kernel을 사용하되 `BLOCK_K`만 `64`로 변경하여 실험하였다.

| Run | Inference Time (ms) | Accuracy (%) | Note |
|---:|---:|---:|---|
| 1 | 239.2161 | 79.83 | Cold run, includes JIT/cache generation |
| 2 | 47.6074 | 79.83 | Cached run |
| 3 | 47.5425 | 79.83 | Cached run |
| 4 | 47.7113 | 79.83 | Cached run |
| 5 | 47.8845 | 79.83 | Cached run |
| 6 | 47.8091 | 79.83 | Cached run |

`BLOCK_K = 64`는 reduction loop의 반복 횟수를 줄일 수 있다는 장점이 있지만, 실제 실험에서는 cold run이 `239.2161 ms`로 증가하여 `200 ms` 기준을 초과하였다. Cached run 역시 약 `47.5-47.9 ms`로, `BLOCK_K = 32`보다 느렸다.

Cached run에서도 `BLOCK_K = 64`가 더 느리다는 점은 이 차이가 단순히 compilation/cache 비용 때문이 아니라 kernel의 런타임 자원 사용 패턴 때문임을 시사한다. `BLOCK_K`를 키우면 한 번의 reduction step에서 처리하는 input/weight tile의 크기(`BLOCK_M * BLOCK_K`, `BLOCK_K * BLOCK_N`)가 커지고, 이를 유지하기 위한 레지스터 사용량도 함께 증가한다. 이로 인해 SM 당 동시에 상주할 수 있는 thread block(또는 warp) 수, 즉 occupancy가 줄어들거나, 레지스터 압박으로 일부 값이 local memory로 spill되어 추가적인 memory traffic이 발생할 수 있다. 또한 tile이 커질수록 mask 계산과 `cks` 좌표 변환(`s`, `r`, `c`)에 사용되는 연산량도 늘어나, reduction loop 반복 횟수 감소로 얻는 이득을 상쇄했을 가능성이 있다. 따라서 최종 제출 후보로는 `BLOCK_K = 32` 설정이 더 안정적이라고 판단하였다.

## 4. Conv2d가 PyTorch보다 비효율적인 이유

이번 Triton 구현은 정확도 기준을 만족했지만, PyTorch baseline과 비교하면 특히 `Conv2d`에서 성능 차이가 발생한다. PyTorch의 `Conv2d`는 일반적으로 `cuDNN`, `CUTLASS` 등 GPU vendor와 framework 차원에서 장기간 최적화된 library kernel을 사용한다. 이러한 library들은 input shape, kernel size, stride, padding, data type 등에 따라 적절한 convolution algorithm을 선택하고, memory access pattern과 Tensor Core 활용까지 정교하게 최적화한다.

반면 이번 과제에서 구현한 Triton `Conv2d`는 하나의 generic direct convolution kernel이다. ResNet18에는 첫 layer의 `7x7` convolution, 대부분의 block에서 사용하는 `3x3` convolution, downsample path의 `1x1` convolution이 함께 존재한다. 하지만 최종 구현은 각 layer에 대해 별도의 특화 kernel이나 알고리즘을 선택하지 않고 동일한 구조의 kernel을 사용한다. 따라서 특정 shape에 최적인 tiling, memory reuse, register usage를 모두 만족시키기 어렵다.

또한 PyTorch library kernel은 convolution을 matrix multiplication으로 변환하거나, implicit GEMM, Winograd, direct convolution 등 여러 알고리즘 중 적절한 방식을 선택할 수 있다. 이번 Triton 구현은 직접 작성한 direct convolution 방식이므로 이러한 알고리즘 선택의 이점을 얻지 못한다. 특히 `Conv2d`는 ResNet18 전체 연산량의 대부분을 차지하므로, Conv2d kernel의 최적화 수준 차이가 전체 inference time에 크게 반영된다.

Memory access 측면에서도 PyTorch의 kernel이 더 유리하다. 고성능 convolution kernel은 shared memory, cache locality, coalesced memory access, Tensor Core-friendly layout 등을 고려해 설계된다. 반면 이번 Triton kernel은 output tile을 기준으로 input과 weight를 load하여 `tl.dot`으로 누적하지만, 모든 layer shape에 대해 최적의 memory reuse를 보장하지는 못한다.

Operator fusion이 없다는 점도 성능 저하의 원인이다. ResNet18에서는 일반적으로 `Conv2d -> BatchNorm2d -> ReLU` 패턴이 반복된다. PyTorch 또는 고성능 inference framework에서는 이러한 연산들을 fuse하여 중간 tensor의 global memory write/read를 줄일 수 있다. 그러나 이번 구현에서는 각 layer가 별도의 Triton kernel로 실행된다. 따라서 `Conv2d` 결과를 global memory에 저장한 뒤, `BatchNorm2d`와 `ReLU`에서 다시 읽고 쓰는 비용이 발생한다.

마지막으로 Triton JIT compilation 비용도 고려해야 한다. 실험 결과에서 `BLOCK_K = 32` 설정의 첫 실행은 `180.7786 ms`였지만, cache된 이후 실행은 약 `43 ms` 수준이었다. 이 차이는 첫 실행에 Triton kernel compilation과 cache generation 시간이 포함되기 때문이다. PyTorch baseline 측정에서는 이미 최적화된 library kernel을 사용하는 경우가 많기 때문에, 직접 작성한 Triton kernel의 cold-start 비용이 상대적으로 크게 나타날 수 있다.

## 5. 결론

이번 과제에서는 ResNet18의 주요 layer를 Triton kernel로 직접 구현하였다. 최종 제출 후보인 single generic Conv2d kernel과 `BLOCK_M = 32`, `BLOCK_N = 32`, `BLOCK_K = 32` 설정에서 accuracy는 `79.83%`로 측정되어 과제 기준인 `75%`를 만족하였다. 또한 Triton cache가 없는 cold run에서도 inference time이 `180.7786 ms`로 측정되어 `200 ms` 기준 안에 들어왔다.

그러나 PyTorch baseline의 inference time인 `28.6428 ms`와 비교하면 Triton 구현은 여전히 느리다. 가장 큰 이유는 PyTorch `Conv2d`가 `cuDNN`과 같은 고도로 최적화된 library kernel을 사용하는 반면, 이번 구현은 하나의 generic direct convolution kernel에 의존하기 때문이다. Layer별 algorithm selection, operator fusion, memory layout 최적화, Tensor Core 활용 측면에서 PyTorch가 더 유리하다.

따라서 이번 구현은 과제의 정확도 및 시간 기준은 만족하지만, PyTorch 수준의 Conv2d 성능을 달성하기 위해서는 layer별 특화 kernel, Conv-BN-ReLU fusion, 더 정교한 tiling 및 memory reuse 최적화가 추가로 필요하다.
