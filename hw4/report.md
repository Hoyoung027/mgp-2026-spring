# CUDA Conv2d Report

---

## Part 1. AI-Generated (Claude)

> Prompt: "CUDA에서 Conv2d를 두 가지 방식으로 구현하는 방법을 설명해줘: (1) Direct convolution, (2) im2col + GEMM. 각각의 원리, 메모리 접근 패턴, 최적화 기법을 2페이지 분량으로 설명해줘."

---

### 1.1 Direct Convolution

Direct convolution은 출력 텐서의 각 원소를 GPU thread 하나가 직접 계산하는 방식이다. 출력 위치 `(oh, ow)`를 담당하는 thread는 입력에서 `kH × kW` 크기의 패치를 읽어 필터와 element-wise 곱셈 후 합산한다.

```
y[k][oh][ow] = Σ_c Σ_kh Σ_kw  x[c][oh + kh][ow + kw] * w[k][c][kh][kw]
```

출력 원소들이 완전히 독립적이므로 "embarrassingly parallel" 구조로, `(outH × outW)` 개의 thread를 동시에 실행하는 것이 자연스럽다. 실제 CUDA 구현에서는 2D thread block (예: 32×8)을 사용하고, grid를 출력 크기에 맞게 설정한다.

**메모리 접근 패턴의 문제**: 인접한 출력 위치를 담당하는 thread들은 입력에서 겹치는 영역을 읽는다. 예를 들어 `(oh=0, ow=0)`과 `(oh=0, ow=1)`은 `kH × (kW - 1)`개의 입력 픽셀을 공유한다. 이를 naive하게 구현하면 global memory 중복 읽기가 발생한다.

**Shared Memory Tiling**: 이 문제를 해결하기 위해 하나의 thread block이 담당하는 출력 타일에 필요한 입력 데이터를 shared memory에 미리 올려두는 tiling 기법을 사용한다. Block 크기가 `TILE_H × TILE_W`라면, `(TILE_H + kH - 1) × (TILE_W + kW - 1)` 크기의 입력 halo 영역을 shared memory에 cooperative하게 로드한다. 이후 각 thread는 global memory 대신 shared memory에서 읽어 연산한다. 3×3 필터 기준 이론적 메모리 절감량은 약 `kH × kW = 9`배이다.

**Constant Memory**: 필터 크기가 작을 경우 (예: 3×3 = 9개, 7×7 = 49개 float), 필터를 `__constant__` 메모리에 올리면 모든 thread에 broadcast 방식으로 전달되어 L1 cache 효율을 높일 수 있다.

---

### 1.2 im2col + GEMM

im2col (image-to-column) 기법은 convolution 연산을 행렬 곱셈(GEMM)으로 변환하는 방법이다. 이를 통해 고도로 최적화된 BLAS 라이브러리를 활용할 수 있다.

**im2col 변환**: 입력 이미지에서 convolution이 참조하는 패치들을 추출해 하나의 2D 행렬로 재배치한다. 출력 위치 `(oh, ow)`에서 필터가 참조하는 `C × kH × kW`개의 픽셀이 col 행렬의 하나의 열(column)이 된다. 최종 col 행렬의 크기는 `(C × kH × kW) × (outH × outW)`이다.

```
col[c * kH * kW + kh * kW + kw][oh * outW + ow] = x[c][oh + kh][ow + kw]
```

각 원소가 독립적으로 계산 가능하므로, col 행렬의 원소 하나를 thread 하나가 담당하는 방식으로 쉽게 병렬화된다. 메모리 coalescing을 위해 column 인덱스(출력 위치)가 thread 인덱스의 빠른 차원이 되도록 배치한다.

**GEMM**: im2col 변환 후 convolution은 다음 행렬 곱셈으로 표현된다:

```
output(K × outH*outW) = filter(K × C*kH*kW) × col(C*kH*kW × outH*outW)
```

이 GEMM은 tiled matrix multiplication으로 최적화된다. `TILE × TILE` 크기의 thread block이 출력 행렬의 타일 하나를 담당하고, A와 B의 타일을 shared memory에 순차적으로 올리며 partial sum을 누적한다. 이를 통해 global memory 접근 횟수를 `TILE`배 줄일 수 있다.

**im2col+GEMM의 장점**: GEMM은 arithmetic intensity가 높고, cuBLAS 같은 라이브러리로 GPU 하드웨어의 Tensor Core를 최대한 활용할 수 있다. C(입력 채널)와 K(출력 채널)가 클수록 GEMM의 행렬 크기가 커지며, 이때 tiling과 cache reuse 효과가 극대화된다. 실제로 PyTorch, cuDNN 등 주요 딥러닝 프레임워크가 im2col+GEMM 또는 그 변형을 기본 convolution 구현으로 채택하는 이유이다.

**float4 Vectorization**: im2col의 경우 연산보다 메모리 대역폭이 병목이므로, `float4` 타입을 이용해 thread 하나가 4개의 원소를 묶어 처리하면 메모리 트랜잭션 수를 줄여 효율을 높일 수 있다.

---

## Part 2. 학생 평가

---

### 2.1 AI 내용 중 올바른 부분

AI가 설명한 이론적 내용은 대체로 정확하다.

- Direct convolution의 thread 매핑 방식과 shared memory tiling의 원리는 실제 구현과 일치한다. "인접한 출력 위치가 입력을 중복으로 읽는다"는 문제 설명이 정확하며, 이것이 tiling의 동기이다.
- im2col 변환의 수식과 col 행렬 크기 `(C × kH × kW) × (outH × outW)`도 맞다. main.cu에서 `col_rows = C * kH * kW`, `col_cols = outH * outW`로 정의한 것과 일치한다.
- float4 vectorization이 메모리 트랜잭션을 줄인다는 설명도 정확하다. 실제로 적용하여 20% 수준의 im2col 시간 감소를 확인했다.

---

### 2.2 AI 내용 중 틀리거나 누락된 부분

AI의 설명은 일반적인 CNN (C=512, K=256 수준)을 가정하고 있어, 이번 과제의 구체적인 조건에는 맞지 않는 부분이 있다.

**"im2col+GEMM이 일반적으로 더 빠르다"는 전제의 한계**

AI는 im2col+GEMM이 BLAS 활용으로 유리하다고 설명했다. 그러나 이번 과제의 입력 조건은 `N=1, C=1, K=1`로 고정되어 있다. 이 경우 im2col+GEMM의 비효율이 두드러진다.

im2col 단계에서 생성되는 중간 행렬의 크기는 다음과 같이 계산된다.

col 행렬의 크기 = `(C × kH × kW) × (outH × outW)` 이며, outH = H - kH + 1 (pad=0, stride=1 기준)이다.

- 3×3 필터:
  - col_rows = C × kH × kW = 1 × 3 × 3 = **9**
  - col_cols = outH × outW = (2048-3+1) × (2048-3+1) = 2046 × 2046 = **4,186,116**
  - 총 원소 = 9 × 4,186,116 = **37,675,044 floats**
  - 메모리 = 37,675,044 × 4 bytes ≈ **144 MB**

- 7×7 필터:
  - col_rows = C × kH × kW = 1 × 7 × 7 = **49**
  - col_cols = outH × outW = (2048-7+1) × (2048-7+1) = 2042 × 2042 = **4,169,764**
  - 총 원소 = 49 × 4,169,764 = **204,318,436 floats**
  - 메모리 = 204,318,436 × 4 bytes ≈ **780 MB**

이 중간 버퍼를 global memory에 쓰고 GEMM 단계에서 다시 읽는 것이 핵심 병목이다. 반면 direct conv는 중간 버퍼 없이 16 MB짜리 입력을 읽어 바로 출력을 계산한다. 실제 측정 결과는 다음과 같다:

| 방식 | 3×3 (avg) | 7×7 (avg) |
|---|---:|---:|
| Direct Conv2d | 0.226 ms | 0.217 ms |
| im2col | 0.429 ms | 1.956 ms |
| GEMM | 0.239 ms | 0.977 ms |
| im2col + GEMM 합계 | 0.668 ms | 2.932 ms |

direct conv가 im2col+GEMM보다 3×3에서 약 3배, 7×7에서 약 14배 빠르다. AI가 "im2col+GEMM이 더 빠를 수 있다"고 설명한 것과 정반대의 결과다.

**M=1 특화 최적화의 효과 차이**

AI는 GEMM tiling을 일반적으로 설명했지만, `K=1` (출력 채널 1개)이면 GEMM의 M=1이 되어 2D tiling의 이점이 사라진다. 이 경우 1D kernel로 단순화했을 때 오히려 더 효율적이다. 실측에서 M=1 특화 커널은 3×3 GEMM에서 21% 개선을 보였으나, 7×7에서는 3%에 그쳤다. 이는 7×7에서 K=49인 reduction dimension이 크지 않아 메모리 대역폭이 여전히 병목이기 때문으로 해석된다.

---

### 2.3 배운 점

**최적화의 일반성과 특수성**

im2col+GEMM이 실제 딥러닝 프레임워크에서 표준으로 쓰이는 이유는 C와 K가 수백 단위일 때 GEMM의 arithmetic intensity가 높아지기 때문이다. 그러나 `C=K=1`에서는 GEMM이 사실상 벡터-행렬 곱에 불과하고, 800 MB 중간 버퍼를 쓰는 비용이 그 이점을 완전히 상쇄한다. 최적화 기법의 효과는 항상 실제 조건에서 검증해야 한다는 것을 실감했다.

**메모리 대역폭이 병목인 경우**

float4 vectorization으로 im2col 시간을 20% 줄였지만, 근본적인 병목인 약 780 MB 쓰기는 사라지지 않는다. GPU의 이론 대역폭이 아무리 높아도, 불필요하게 큰 중간 버퍼를 만드는 알고리즘 자체의 한계가 더 크다. 알고리즘 선택이 커널 최적화보다 먼저라는 교훈을 얻었다.

**Naive 구현으로도 충분한 경우**

제한 시간 기준(3×3: 0.8초, 7×7: 6.0초)에 비해 실제 측정값은 수백~수천 배 여유가 있다. 이는 4.18M개의 독립적인 출력 원소를 현대 GPU가 병렬로 처리하기에 연산량 자체가 매우 작기 때문이다. 복잡한 최적화보다 올바른 병렬화 구조가 우선임을 확인했다.
