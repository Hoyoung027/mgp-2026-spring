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

### 1.3 현재 과제 조건에 대한 구현 및 성능 예상

제공된 driver에서 batch size는 1이고, 입력은 단일 채널 2048×2048 이미지이며 필터는 3×3과 7×7 두 가지이다. 이 경우 direct convolution은 출력 원소 하나를 thread 하나가 담당하는 단순한 mapping만으로도 충분한 병렬성을 확보할 수 있다. 출력 원소 수가 약 4백만 개이므로 GPU 전체를 활용하기에 충분하고, 각 thread가 수행하는 연산량은 3×3에서 9번, 7×7에서 49번의 multiply-add로 작다.

im2col 구현에서는 col 행렬의 column 방향을 연속 메모리로 배치하는 것이 중요하다. 특히 `outH × outW` 방향이 연속되므로, thread 하나가 연속된 4개 원소를 처리하는 `float4` vectorization을 적용하면 store instruction 수를 줄일 수 있다. GEMM의 경우 출력 채널 `K`가 1이라면 결과 행렬의 row 수가 1이 되므로, 일반적인 2D tiled GEMM보다 row 하나에 특화한 1D kernel이 더 단순하고 효율적일 수 있다.

성능 면에서는 direct convolution과 im2col+GEMM 모두 제한 시간 안에 들어올 것으로 예상된다. 다만 일반적인 CNN에서는 im2col+GEMM이 유리한 경우가 많기 때문에, 적절히 최적화한 GEMM을 사용하면 im2col+GEMM 방식도 direct convolution과 비슷하거나 더 빠른 성능을 낼 가능성이 있다. 특히 7×7처럼 필터가 커질수록 GEMM으로 변환했을 때 얻는 계산 구조의 장점이 커질 수 있다.

---

## Part 2. 학생 평가

---

### 2.1 AI 내용 중 올바른 부분

AI가 설명한 이론적 내용은 대체로 정확하다.

- Direct convolution의 thread 매핑 방식과 shared memory tiling의 원리는 실제 구현과 일치한다. "인접한 출력 위치가 입력을 중복으로 읽는다"는 문제 설명이 정확하며, 이것이 tiling의 동기이다.
- im2col 변환의 수식과 col 행렬 크기 `(C × kH × kW) × (outH × outW)`도 맞다. main.cu에서 `col_rows = C * kH * kW`, `col_cols = outH * outW`로 정의한 것과 일치한다.
- float4 vectorization이 메모리 트랜잭션을 줄인다는 설명도 정확하다. 실제로 적용하여 20% 수준의 im2col 시간 감소를 확인했다.
- 제공된 driver에서 convolution의 출력 채널 수가 1일 때 GEMM 결과 행렬의 row 수가 `M=1`이 되므로, 이를 1D kernel로 특화할 수 있다는 지적도 유효했다. 실제로 3×3 GEMM에서는 naive 2D GEMM보다 약 21% 빠른 결과가 나왔다.

---

### 2.2 AI 내용 중 틀리거나 누락된 부분

AI의 설명은 입력 채널 C와 출력 채널 K가 충분히 큰 일반적인 CNN convolution을 기준으로 한 설명에 가깝다. 따라서 제공된 `main.cu`의 고정 설정에서 관찰된 병목을 충분히 예측하지 못한 부분이 있다.

**"im2col+GEMM이 일반적으로 더 빠르다"는 전제의 한계**

AI는 im2col+GEMM이 BLAS 활용으로 유리하다고 설명했고, 7×7처럼 필터가 커질수록 GEMM 방식의 장점이 커질 수 있다고 예측했다. 그러나 제공된 `main.cu`에서는 `N=1, C=1, K=1`로 설정되어 있고, 이 조건에서는 im2col+GEMM의 비효율이 두드러진다.

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

direct conv가 im2col+GEMM보다 3×3에서 약 3배, 7×7에서 약 14배 빠르다. AI의 예상과 달리, 필터가 7×7로 커졌을 때도 GEMM 구조의 이점보다 im2col 중간 버퍼를 만들고 다시 읽는 비용이 훨씬 크게 작용했다.

최적화 효과를 확인하기 위해 naive im2col 및 naive 2D GEMM을 사용한 baseline도 측정했다. 평균 시간은 다음과 같다.

| 구현 | 3×3 im2col | 3×3 GEMM | 3×3 im2col+GEMM | 7×7 im2col | 7×7 GEMM | 7×7 im2col+GEMM |
|---|---:|---:|---:|---:|---:|---:|
| Naive baseline | 0.536 ms | 0.304 ms | 0.841 ms | 2.486 ms | 1.011 ms | 3.498 ms |
| Optimized | 0.429 ms | 0.239 ms | 0.668 ms | 1.956 ms | 0.977 ms | 2.932 ms |

이 표에서 볼 수 있듯이 `float4` im2col과 M=1 특화 GEMM은 naive baseline보다 빠르다. 그러나 최적화 후에도 im2col+GEMM 전체 시간은 direct convolution보다 크다. 따라서 문제는 단순히 커널 최적화가 부족한 것이 아니라, im2col 중간 버퍼를 materialize하는 알고리즘적 비용이 크다는 점에 있다.

**M=1 특화 최적화의 효과 차이**

AI는 GEMM tiling을 일반적으로 설명했지만, convolution의 출력 채널 수가 1이면 GEMM 결과 행렬의 row 수가 `M=1`이 되어 2D tiling의 이점이 작아진다. 이 경우 1D kernel로 단순화했을 때 오히려 더 효율적이라고 보인다. 이 성능 향상은 naive 2D GEMM 평균 시간과 M=1 특화 GEMM 평균 시간을 비교하여 계산했다.

| 필터 | naive GEMM avg | M=1 GEMM avg | 개선율 |
|---|---:|---:|---:|
| 3×3 | 0.304 ms | 0.239 ms | 약 21% |
| 7×7 | 1.011 ms | 0.977 ms | 약 3% |

개선율은 `(naive time - optimized time) / naive time`으로 계산했다. 3×3에서는 기존 2D block에서 실제로 필요한 row가 하나뿐이라 낭비되는 thread가 많았고, 1D kernel이 이 낭비를 줄였다. 반면 7×7에서는 reduction dimension이 `C×kH×kW = 49`로 더 커져 각 output column이 수행해야 하는 dot product 자체의 비용이 커졌기 때문에, row 방향 thread 낭비를 줄이는 효과가 상대적으로 작게 나타났다.

---

### 2.3 배운 점

**최적화의 일반성과 특수성**

im2col+GEMM이 실제 딥러닝 프레임워크에서 표준으로 쓰이는 이유는 C와 K가 수백 단위일 때 GEMM의 arithmetic intensity가 높아지기 때문이다. 그러나 `C=K=1`에서는 GEMM이 사실상 벡터-행렬 곱에 불과하고, 800 MB 중간 버퍼를 쓰는 비용이 그 이점을 완전히 상쇄한다. 최적화 기법의 효과는 항상 실제 조건에서 검증해야 한다는 것을 실감했다.

**메모리 대역폭이 병목인 경우**

float4 vectorization으로 im2col 시간을 20% 줄였지만, 근본적인 병목인 약 780 MB 쓰기는 사라지지 않는다. GPU의 이론 대역폭이 아무리 높아도, 불필요하게 큰 중간 버퍼를 만드는 알고리즘 자체의 한계가 더 크다. 알고리즘 선택이 커널 최적화보다 먼저라는 교훈을 얻었다.

**Naive 구현으로도 충분한 경우**

제한 시간 기준(3×3: 0.8초, 7×7: 6.0초)에 비해 실제 측정값은 수백~수천 배 여유가 있다. 이는 4.18M개의 독립적인 출력 원소를 현대 GPU가 병렬로 처리하기에 연산량 자체가 매우 작기 때문이다. 복잡한 최적화보다 올바른 병렬화 구조가 우선임을 확인했다.
