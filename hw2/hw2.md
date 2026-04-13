# HW2 구현 설명

## 개요

`parallel.h`에 세 함수를 구현했다. `<thread>`와 `<vector>` 헤더를 추가하여 C++ 표준 스레드를 사용했으며, OpenMP는 사용하지 않았다.

---

## init_vec

```cpp
srand(42);
for (int i = 0; i < N; ++i)
    a[i] = (float)rand() / (float)RAND_MAX;
```

Freivalds 알고리즘에서 사용할 랜덤 벡터를 초기화한다. 이 함수는 타이머 밖에서 호출되므로 성능보다 정확성이 중요하다. 고정 시드(42)를 사용하여 재현 가능한 값을 생성한다.

---

## gemv (General Matrix-Vector Multiplication)

### 구현 방식

행 단위 병렬 분할. N개의 행을 스레드 수만큼 균등하게 나눠서 각 스레드가 독립적으로 담당 행을 계산한다.

```
스레드 0: 행 0    ~ 행 255  담당
스레드 1: 행 256  ~ 행 511  담당
...
스레드 T-1: 행 ~  ~ 행 2047 담당 (나머지 포함)
```

각 스레드가 계산하는 내용:
```
for i in [i_start, i_end):
    c[i] = a[i][0]*b[0] + a[i][1]*b[1] + ... + a[i][N-1]*b[N-1]
```

### v1: 8-way Scalar Unrolling

각 행의 dot product 계산 시 8개의 독립된 누산기(`sum0`~`sum7`)를 사용해 8-way unrolling을 적용했다.

```cpp
float sum0=0, sum1=0, sum2=0, sum3=0, sum4=0, sum5=0, sum6=0, sum7=0;
for (; k <= N - 8; k += 8) {
    sum0 += ai[k]   * b[k];
    sum1 += ai[k+1] * b[k+1];
    sum2 += ai[k+2] * b[k+2];
    sum3 += ai[k+3] * b[k+3];
    sum4 += ai[k+4] * b[k+4];
    sum5 += ai[k+5] * b[k+5];
    sum6 += ai[k+6] * b[k+6];
    sum7 += ai[k+7] * b[k+7];
}
float sum = (sum0+sum1)+(sum2+sum3)+(sum4+sum5)+(sum6+sum7);
```

단순 `sum += ai[k]*b[k]` 는 이전 누적 결과에 의존하므로 CPU가 직렬로 처리한다. 8개의 독립된 누산기를 쓰면 각 FMA 연산이 이전 결과를 기다리지 않아 Instruction-Level Parallelism (ILP)을 활용할 수 있다. 다만 컴파일러 최적화 플래그 없이는 실제 SIMD 명령어로 변환되지 않아 스칼라 연산으로만 동작한다.

### v2: SSE2 Intrinsics

`<immintrin.h>`를 추가하여 SSE2 intrinsic으로 내부 루프를 교체했다. SSE2는 x86-64의 기본 베이스라인이므로 컴파일 플래그나 `pragma`, `__attribute__` 없이도 동작한다.

```cpp
__m128 vsum0 = _mm_setzero_ps();
__m128 vsum1 = _mm_setzero_ps();

for (; k <= N - 8; k += 8) {
    vsum0 = _mm_add_ps(vsum0, _mm_mul_ps(
        _mm_loadu_ps(ai + k),
        _mm_loadu_ps(b  + k)));
    vsum1 = _mm_add_ps(vsum1, _mm_mul_ps(
        _mm_loadu_ps(ai + k + 4),
        _mm_loadu_ps(b  + k + 4)));
}

// horizontal reduction: [v0,v1,v2,v3] -> v0+v1+v2+v3
__m128 vsum = _mm_add_ps(vsum0, vsum1);
__m128 tmp  = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(1,0,3,2));
vsum = _mm_add_ps(vsum, tmp);
tmp  = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2,3,0,1));
vsum = _mm_add_ps(vsum, tmp);
float sum = _mm_cvtss_f32(vsum);
```

`_mm_loadu_ps`로 4개의 float를 한 번에 로드하고, `_mm_mul_ps` + `_mm_add_ps`로 4-wide 병렬 연산을 수행한다. 2개의 `__m128` 누산기(`vsum0`, `vsum1`)를 사용해 이터레이션당 8 floats를 처리하면서 누산기 간 의존성도 제거했다. 루프 종료 후 shuffle 기반 수평 합산(horizontal reduction)으로 4개 레인의 합을 하나의 scalar로 추출한다.

### Exact Float Match 보장

`c[i]`를 계산할 때 k=0부터 N-1까지 순서대로 누적한다. 각 행의 계산이 독립적이고 k 순서가 레퍼런스와 동일하므로 부동소수점 연산 결과가 정확히 일치한다.

### 스레드 생성

```cpp
int num_threads = std::thread::hardware_concurrency();
std::vector<std::thread> threads;
for (int tid = 0; tid < num_threads; ++tid)
    threads.emplace_back([=]() { /* 담당 행 계산 */ });
for (auto &t : threads) t.join();
```

`hardware_concurrency()`로 서버 코어 수에 맞게 자동 결정한다.

---

## gemm (General Matrix-Matrix Multiplication)

### 구현 방식

행 밴드 병렬 분할 + 3중 캐시 블로킹(Tiling).

**병렬화:** gemv와 동일하게 행 범위를 스레드별로 나눈다. 각 스레드는 자신이 담당하는 행 범위의 출력 행렬 C를 독립적으로 계산한다.

**캐시 블로킹:** TILE=64로 i, k, j 세 방향을 타일링한다.

### 루프 구조

```
각 스레드 (i 범위: i_start ~ i_end):

  1. 담당 행 c 초기화
     for i in [i_start, i_end): c[i][0..N] = 0

  2. 타일링 ikj 순서 GEMM
     for ii in [i_start, i_end) step 64:
       for kk in 0..N step 64:
         for jj in 0..N step 64:
           for i in ii..ii+64:
             for k in kk..kk+64:
               a_ik = a[i][k]         ← 레지스터에 고정
               for j in jj..jj+64:
                 c[i][j] += a_ik * b[k][j]
```

### TILE=64를 선택한 이유

| 항목 | 계산 |
|---|---|
| 타일 하나의 크기 | 64 × 64 × 4 bytes = 16 KB |
| A + B + C 타일 합계 | 48 KB |
| L2 캐시 (일반적) | 256 KB |

세 타일이 L2 캐시 안에 상주하여, `kk` 루프 동안 A 블록이 캐시에 남아 `jj` 반복마다 재사용된다. `j` 방향으로 `b[k][j]`와 `c[i][j]`를 연속 접근하므로 캐시 라인 활용률이 높다.

### Exact Float Match 보장

`kk` 루프가 0, 64, 128, ... 순서로 진행되므로 각 `c[i][j]`에 대한 k 누적 순서는 0, 1, 2, ..., N-1로 레퍼런스 serial 구현과 동일하다.

---

## 성능 특성

| 항목 | 내용 |
|---|---|
| 스레드 수 | `hardware_concurrency()` 자동 감지 |
| gemv 스레드 생성 | 호출마다 생성/join (Freivalds에서 3회) |
| gemm 스레드 생성 | 호출마다 1회 생성/join |
| 캐시 최적화 | gemm에 TILE=64 3중 블로킹 적용 |
| OpenMP | 미사용 |
| 전역 변수 | 미사용 (모두 TODO 영역 안에서 처리) |
