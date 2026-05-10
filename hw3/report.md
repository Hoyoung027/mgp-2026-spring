# HW3: CUDA LoRA Implementation Report

**학번-이름**: [학번]-[이름]  
**GPU**: NVIDIA GeForce RTX 3090 (sm_86)  
**빌드**: `nvcc -std=c++11 -arch=sm_86`  
**문제 크기**: B=32, in_dim=4096, out_dim=4096, r=8, alpha=16

---

## Pages 1–2: AI-Written Section

### 1. 병렬 알고리즘 구현

#### 1.1 LoRA 연산 정의

LoRA는 아래 수식으로 정의된다.

```
y = x @ W.T + scale * (x @ A.T @ B.T)
    [B, out] = [B, K] @ [K, out]  +  scale * ([B, K] @ [K, r] @ [r, out])
```

이 중 `x @ A.T`는 소규모 행렬 곱 ([32, 4096] @ [4096, 8])이고, `x @ W.T`는 대규모 행렬 곱 ([32, 4096] @ [4096, 4096])이다. 두 연산의 규모 차이가 크기 때문에 단일 커널보다 두 커널로 분리하는 것이 효율적이다.

#### 1.2 2-커널 설계

**Kernel 1 — `kernel_xA`**: `xA = x @ A.T` 계산

- grid: `(B=32, r=8)`, block: 32 threads
- 한 블록이 `xA[b, ri]` 하나를 담당한다.
- 4096-wide dot product를 32 threads가 나눠 처리한 뒤 shared memory reduction으로 합산한다.
- `partial[32]` 공유 메모리에 partial sum을 저장하고 tree reduction으로 합친다.

**Kernel 2 — `kernel_xWT_fused`**: `y = x @ W.T + scale * xA @ B.T` fused 계산

- grid: `(out_dim/BLK_X, B/BLK_Y)` = `(256, 2)`, block: `(THREADS_X=8, BLK_Y=16)`
- 두 shared memory tile을 사용한다.
  - `sX[BLK_Y][TILE_K]`: x의 K-방향 슬라이스
  - `sW[BLK_X][TILE_K+1]`: W의 K-방향 슬라이스 (+1 padding)
  - `sXA[BLK_Y][8]`: xA 캐시 (같은 블록 내 모든 thread가 공유)
- 각 thread가 `OUTS_PER_THREAD=2`개의 출력 column을 동시에 계산한다.
- K 루프 마지막에 LoRA 항(`scale * xA @ B.T`)을 추가 global memory round-trip 없이 fused store로 처리한다.

#### 1.3 주요 최적화 요약

| 최적화 | 적용 위치 | 효과 |
|--------|-----------|------|
| float4 벡터화 로드 | sX, sW 로드 구간 | global load instruction 수 1/4 감소 |
| TILE_K=128 | K-loop | sync 횟수 32회, cache 재사용 개선 |
| OUTS_PER_THREAD=2 | 계산 루프 | sX 값 2회 재사용, ILP 향상 |
| sW +1 padding | `sW[BLK_X][TILE_K+1]` | sW 접근 bank conflict 제거 |
| sXA shared cache | LoRA tail | xA global load를 1회로 압축 |
| B_mat float4 로드 | LoRA tail | r=8 scalar 8회 → float4 2회 |
| kernel_xA block reduction | kernel_xA | 4096 dot product latency 대폭 감소 |
| `__launch_bounds__(128, 4)` | kernel_xWT_fused | SM당 4 blocks 목표로 register allocation 조정 |

---

### 2. 성능 분석 및 마진 예측

#### 2.1 단계별 측정 결과

아래 표는 RTX 3090에서 5회 반복 측정한 결과다. 채점은 **5회 중 최대값** 기준이므로 worst-case가 핵심이다.

| 버전 | best (ms) | worst (ms) | avg (ms) | 최대 오차 |
|------|----------:|-----------:|---------:|----------:|
| 1-thread-1-output (초기) | 0.714 | 0.727 | 0.720 | 0.005859 |
| + OUTS_PER_THREAD=2, float4, fused LoRA | 0.674 | 0.684 | 0.679 | 0.005859 |
| + kernel_xA block reduction (BLK_X=32, BLK_Y=8) | 0.573 | 0.579 | 0.576 | 0.007324 |
| + BLK_X=16, BLK_Y=16 + `__launch_bounds__` | **0.548** | **0.562** | **0.555** | 0.007324 |

worst-case 기준으로 초기 대비 **약 22.7%** 개선되었다.

#### 2.2 이론적 Roofline 분석

RTX 3090 스펙: FP32 peak 35.58 TFLOPS, 메모리 대역폭 936 GB/s.

메인 연산 `x @ W.T` ([32, 4096] × [4096, 4096]) 기준:

- FLOPs: 2 × 32 × 4096 × 4096 ≈ 1.07 GFLOP
- 메모리 (W 전체): 4096 × 4096 × 4 ≈ 64 MB
- 산술 집약도: 1.07 GFLOP / 64 MB ≈ **16.6 FLOP/byte**
- Ridge point: 35,580 / 936 ≈ 38 FLOP/byte → **메모리 바운드** 구간

메모리 바운드 상한 시간: 64 MB / 936 GB/s ≈ **0.068 ms**  
현재 실측: 0.55 ms → 이론 대역폭 대비 약 **12.5% 활용** 수준

이 차이는 단순히 대역폭이 낮아서가 아니라, 작은 batch (B=32)로 인해 512개 블록만 생성되어 RTX 3090의 82개 SM을 충분히 채우지 못하는 **occupancy 제약**과, 아래에서 설명하는 **shared memory bank conflict**에서 비롯된다.

#### 2.3 sX 패딩 누락에 의한 Bank Conflict 예측

현재 코드에서 `sW`는 `[BLK_X][TILE_K+1]`로 +1 패딩이 적용되어 있으나, `sX`는 `[BLK_Y][TILE_K]`로 패딩이 없다.

```cpp
__shared__ float sX[BLK_Y][TILE_K];       // ← 패딩 없음
__shared__ float sW[BLK_X][TILE_K + 1];   // ← +1 패딩 적용
```

내부 계산 루프는 다음과 같다.

```cpp
for (int tk = 0; tk < TILE_K; tk++) {
    float xv = sX[ty][tk];  // ← 이 접근이 문제
    for (int j = 0; j < OUTS_PER_THREAD; j++)
        val[j] += xv * sW[tx + j * THREADS_X][tk];
}
```

CUDA shared memory는 32개의 bank로 구성되며, `float` 기준으로 주소 `a`는 bank `a % 32`에 속한다. `sX[BLK_Y][TILE_K]`의 row stride는 128 floats이고 128 % 32 = 0이므로, 모든 row가 동일한 bank layout을 갖는다.

2D thread block `(THREADS_X=8, BLK_Y=16)`에서 warp는 tid로 선형화된다. Warp 0은 ty=0~3, tx=0~7의 32개 thread를 포함한다. 이 warp가 `sX[ty][tk]`에 접근할 때:

| ty | 주소 오프셋 | bank |
|----|------------|------|
| 0 | `0×128 + tk` | `tk % 32` |
| 1 | `1×128 + tk` | `tk % 32` |
| 2 | `2×128 + tk` | `tk % 32` |
| 3 | `3×128 + tk` | `tk % 32` |

4개의 서로 다른 주소가 **모두 같은 bank**에 속한다 → **4-way bank conflict** 발생.

이를 해결하려면 `sX`에도 +1 패딩을 추가한다.

```cpp
__shared__ float sX[BLK_Y][TILE_K + 1];   // +1 패딩 추가
```

패딩 후 row stride는 129 floats이며, gcd(129, 32) = 1이므로 서로 다른 row는 서로 다른 bank에 접근한다.

| ty | bank |
|----|------|
| 0 | `tk % 32` |
| 1 | `(tk + 1) % 32` |
| 2 | `(tk + 2) % 32` |
| 3 | `(tk + 3) % 32` |

bank conflict 해소 → **이론상 내부 루프의 shared memory 처리량이 최대 4배 개선 가능**.

내부 루프는 전체 커널 시간의 대부분을 차지하므로, 이 수정이 현재 worst-case 0.562 ms를 **0.4~0.5 ms 수준으로 개선**할 것으로 예측한다.

---

## Pages 3–4: 본인 평가 (User Evaluation)

### 3. AI 내용 검증: 패딩 실험 결과

#### 3.1 실험 설계

AI의 2.3절 예측을 검증하기 위해 두 가지 실험을 진행했다.

**실험 A — sX 패딩 추가**: AI가 제안한 대로 `sX`에 +1 패딩 추가

```cpp
// 수정 전
__shared__ float sX[BLK_Y][TILE_K];

// 수정 후
__shared__ float sX[BLK_Y][TILE_K + 1];
```

**실험 B — sW 패딩 제거**: 대조 실험으로, 이미 적용된 `sW` 패딩을 제거

```cpp
// 수정 전
__shared__ float sW[BLK_X][TILE_K + 1];

// 수정 후
__shared__ float sW[BLK_X][TILE_K];
```

두 실험 모두 2D 인덱스(`sX[row][col]`, `sW[row][col]`)는 그대로이며, 패딩 유무만 달리하여 물리적 stride만 변경된다.

#### 3.2 실험 결과

| 버전 | best (ms) | worst (ms) | avg (ms) | 최대 오차 |
|------|----------:|-----------:|---------:|----------:|
| baseline (현재 코드) | 0.556 | 0.566 | 0.561 | 0.007324 |
| 실험 A: sX +1 패딩 추가 | 0.669 | 0.689 | 0.673 | 0.007324 |
| 실험 B: sW 패딩 제거 | 1.377 | 1.402 | 1.390 | 0.007324 |

각 버전은 RTX 3090에서 5회 반복 측정하였다.

#### 3.3 AI 예측 평가

**맞은 점**

AI는 `sW`에는 패딩이 있고 `sX`에는 없다는 코드 상의 비대칭을 정확하게 짚었다. Bank conflict 발생 조건(row stride % 32 == 0)과 gcd(129, 32) = 1에 의한 해소 원리는 이론적으로 옳다. 실험 B에서 `sW` 패딩 제거 시 2.5배 느려진 것은 AI의 bank conflict 분석이 `sW`에 대해서는 정확했음을 입증한다.

**틀린 점**

AI는 `sX`에 패딩을 추가하면 "이론상 내부 루프의 처리량이 최대 4배 개선 가능"하며 "0.4~0.5 ms 수준으로 개선될 것"이라고 예측했다. 그러나 실험 A의 결과는 0.561 ms → 0.673 ms로 오히려 **20% 성능이 하락**했다. 예측이 완전히 틀렸다.

**원인 분석: sX와 sW의 차이**

`sW`와 `sX`는 둘 다 같은 bank conflict 조건을 갖지만, 계산 루프에서의 접근 패턴이 본질적으로 다르다.

`sW[tx + j*THREADS_X][tk]` 접근에서 warp 내 tx = 0~7은 **8개의 서로 다른 주소**를 읽는다. 각 tx마다 실질적으로 다른 데이터가 필요하므로, 같은 bank에 몰리면 8번 직렬화(8-way conflict)가 불가피하다.

반면 `sX[ty][tk]` 접근에서 같은 ty를 공유하는 8개의 tx 스레드는 **동일한 주소**를 읽는다. GPU는 동일 주소를 읽는 여러 스레드를 broadcast로 처리하여 bank conflict 없이 1 transaction으로 해결한다. 따라서 4개의 ty 그룹 간의 conflict는 존재하지만, Ampere의 warp scheduler가 다른 warp 실행으로 이 latency를 충분히 숨기고 있다.

패딩을 추가하면 row stride가 128(2의 거듭제곱)에서 129로 바뀐다. 컴파일러가 shift 연산 대신 일반 곱셈을 사용하게 되고, 이로 인한 주소 계산 오버헤드와 shared memory layout 변화가 broadcast 이득보다 크게 작용한 것으로 판단된다.

**빠진 점**

AI는 `sX`와 `sW`의 접근 패턴 차이(broadcast vs distinct)를 구분하지 않고 동일하게 취급했다. Bank conflict의 유무만 보고 broadcast가 이미 제공하는 효율성을 간과한 것이 예측 실패의 근본 원인이다.

---

### 4. 배운 것과 최종 평가

#### 4.1 실험을 통해 배운 것

- **이론과 실측의 괴리**: Roofline 모델 상한이 0.068 ms인데 실측이 0.56 ms인 것은 단순히 대역폭 미활용이 아니라, 낮은 batch로 인한 block 부족, sync overhead, bank conflict가 복합적으로 작용하기 때문이다.

- **최악 성능 안정성의 중요성**: 채점이 5회 최대값 기준이므로 최고 기록보다 흔들리지 않는 안정성이 더 중요했다. `__launch_bounds__`는 peak를 높이기보다 worst-case 분산을 줄이는 데 기여했다.

- **"이론상 좋아 보이는 것"이 느릴 수 있다**: `TILE_K=256`은 sync 횟수를 줄이는 이론적 장점이 있었지만 shared memory 증가로 occupancy가 낮아져 오히려 0.795 ms로 나빠졌다. sX 패딩도 마찬가지였다.

- **Bank conflict는 접근 패턴과 함께 분석해야 한다**: Bank conflict 이론은 맞지만, broadcast가 이미 작동하는 경우에는 conflict가 실질적인 병목이 아닐 수 있다. 반대로 `sW`처럼 8개의 고유 주소가 같은 bank에 몰리는 경우는 2.5배 저하로 직결됐다. 같은 bank conflict라도 접근 패턴에 따라 영향이 크게 다름을 실험으로 확인했다.

#### 4.2 최종 성능 및 남은 마진

| 항목 | 값 |
|------|-----|
| 최종 worst-case | 0.566 ms |
| 최대 절대 오차 | 0.007324 |
| 이론 상한 (roofline) | ~0.068 ms |
| 이론 대비 달성률 | ~12% |

현재 구현에서 남아 있는 잠재적 마진:

1. **Block count 증가**: B=32로 인해 512 블록 한계. batch 방향 parallelism이 근본적으로 부족하다.
2. **cuBLAS 대비 격차**: cuBLAS는 tensor core 기반 WMMA를 사용하며, 같은 하드웨어에서 현재 구현보다 빠를 것으로 예상된다. 수동 FP32 커널의 한계가 있다.
3. **double buffering**: `cp.async`와 double-buffered shared memory로 K-loop의 load latency를 숨길 수 있다.

이 중 1번은 문제 크기가 고정이므로 해결이 불가능하고, 2번은 구현 복잡도가 높다. 3번은 시간 제약상 다음 최우선 실험 후보다.
