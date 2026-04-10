# HW2 Report Plan: Matrix Verification Challenge

> **형식**: AI 생성 2페이지 + 본인 평가 2페이지 (총 4페이지)  
> **파일명**: `학번-이름.pdf`  
> 아래 내용을 Claude/GPT에 입력해 AI 파트를 생성하고, 이후 본인 평가 파트를 직접 작성

---

## Page 1-2: AI 생성 파트

### AI에게 줄 프롬프트

```
Write a 2-page academic report for a Multicore and GPU Programming (MGP) course assignment.
The topic is parallel implementations of GEMV and GEMM for N=2048 matrices, and an experiment
proving that thread creation cost is the dominant bottleneck in GEMV.

Include the following sections:

1. Implementation Overview
2. Experimental Evidence: Thread Creation is the Bottleneck
3. Freivalds' Algorithm: Why It Is Probabilistic
4. Performance Analysis and Margin
5. Results Table

Details to include:

--- Implementation ---
- GEMM: ikj cache tiling (TILE=64), multi-threading via hardware_concurrency(),
  SSE2 j-loop with 16-wide unrolling (_mm_set1_ps broadcast + _mm_mul_ps/_mm_add_ps).
  Result: ~0.488s (bonus threshold: 1.9s).

- GEMV (submitted version, hw2/parallel.h): per-call thread creation, SSE2 with
  2 accumulators per row (8 floats/iter). Result: ~0.0097s for Freivalds.

- GEMV (experimental version): static thread pool of 16 threads with spin+yield
  synchronization (work_ver atomic counter), SSE2 4-row × 2-accumulator kernel.
  Result: ~0.0016-0.002s for Freivalds.

--- Thread Creation Experiment (experiment_threadpool) ---
Four conditions were measured to isolate thread creation overhead:

A. Thread creation cost only (no compute):
   - Measured: creating and joining N threads with no work
   - Shows per-thread creation latency (~50us/thread on this server)

B. Per-call thread creation × 3 gemv calls (baseline hw2/parallel.h approach):
   - 16 threads created, work done, joined — repeated 3 times
   - Total = thread_creation×3 + compute×3

C. Thread pool created INSIDE Freivalds timer:
   - Pool of 16 threads created once, 3 gemv calls reuse it
   - Total = pool_creation + compute×3

D. Thread pool created OUTSIDE Freivalds timer (pure compute):
   - Pool pre-created before timer starts, only compute is timed
   - Total = compute×3 only

Key equations:
  C - D = pool creation cost
  B - D = per-call thread creation overhead × 3
  (B - D) / 3 ≈ A result for 15 threads → confirms thread creation is the bottleneck

--- Freivalds' Algorithm ---
- Probabilistic verification: A×B=C verified via A(Bv)=Cv for random vector v
- O(N²) per round vs O(N³) for full GEMM → ~20x faster for N=2048
- Error probability: if AB≠C, probability of false acceptance ≤ 1/2 per round
- With 3 rounds (as implemented): error probability ≤ (1/2)³ = 12.5%
- In GF(2) field: bound is tight; in floating point, numerical errors add complexity

--- SSE2 SIMD ---
- 128-bit XMM registers: 4 floats processed simultaneously
- GEMM: _mm_set1_ps(a[i][k]) broadcasts scalar, then 16-wide j-loop processes b row
- GEMV: 4-row kernel loads b[k..k+7] once, reuses for 4 rows simultaneously;
  2 accumulators per row hide 4-cycle _mm_add_ps latency (ILP doubling)

--- Results Table (fill in measured values) ---
| Experiment | Configuration | Freivalds Time |
|------------|--------------|----------------|
| A | 15 threads creation only | ___ ms |
| B | Per-call 16 threads × 3 | ___ ms |
| C | Pool inside timer | ___ ms |
| D | Pool outside timer (compute only) | ___ ms |
| D×3 | Theoretical minimum (3× single gemv compute) | ___ ms |
| hw2/parallel.h | Per-call threads, SSE2 2-acc | ~9.7 ms |
| experiment | Pool inside timer, SSE2 4-row 2-acc | ~1.7 ms |
```

---

## Page 3-4: 본인 평가 파트 (직접 작성)

### 3-1. AI가 맞게 설명한 것

- **ikj 루프 순서**: B 행렬을 j 방향으로 sequential 접근하므로 cache line 재사용 효율이 높다는 설명은 정확
- **Freivalds 오차 확률 (1/2)^k**: GF(2) 위에서의 bound는 수학적으로 엄밀히 증명된 내용
- **Thread pool의 이점**: 생성 비용 제거로 재사용 시 overhead가 줄어든다는 논리는 맞음
- **SSE2 설명**: `_mm_set1_ps` broadcast + `_mm_mul_ps`의 동작 설명은 정확

### 3-2. AI가 틀리거나 과장한 것

- **"Thread pool이 항상 빠르다"**: 실험 C와 D의 차이에서 보듯, pool 생성 비용이 타이머 안에 포함되면 (C 조건) 이점이 절반으로 줄어든다. AI는 이 구조적 한계를 단순화함
- **"16 threads가 최적"**: 이론적으로는 더 많은 스레드가 유리하지만, 실험 A에서 보듯 thread 생성 비용이 compute 비용을 초과하는 구간이 존재함. 최적은 실험으로만 확인 가능
- **AVX2 적용 제안**: Makefile을 수정할 수 없는 제약 조건에서는 불가능한 내용을 성능 개선 방향으로 언급할 수 있음

### 3-3. AI가 누락한 것

- **"maximum of 5 executions" 정책**: 채점이 5회 중 최악 기준이므로, 분산이 큰 구현(C 조건의 0.0016-0.002s)은 worst case로 채점될 수 있음. 안정성이 성능만큼 중요하다는 점을 AI는 언급하지 않음
- **Pool 생성 위치의 구조적 제약**: `static` 변수는 첫 번째 함수 호출에 초기화되므로, 타이머 안에 포함될 수밖에 없다는 C++ 언어 레벨의 제약
- **spin+yield 하이브리드의 필요성**: 순수 spinlock은 CPU 주파수 throttling을 유발하고, 순수 yield는 OS 스케줄러 latency를 추가함. 8000회 pause → yield 하이브리드는 실험으로 도출한 heuristic임

### 3-4. 실험에서 직접 배운 것 (experiment_threadpool 결과 기반)

> 아래 괄호를 실제 측정값으로 채울 것

- 실험 A: 스레드 1개 생성 비용 ≈ (___)μs
  → 15개 스레드 pool 생성 ≈ (___)ms → 이것이 Freivalds 타이머에서 지배적 비용임을 수치로 확인
- 실험 B vs D: per-call 방식의 thread 생성 overhead = (___)ms
  → 순수 compute 대비 (___)배의 overhead
- 실험 C vs D: `static` pool 방식에서 pool 생성 비용 = (___)ms
  → 이를 타이머 밖으로 빼면 Freivalds 시간이 (___)ms로 단축됨
- **결론**: Thread 생성이 병목이며, 이를 타이머 밖으로 이동하는 것이 가장 효과적인 최적화임.
  hw2/parallel.h의 `static GemvPool pool(16)` 방식은 이를 부분적으로 해결하지만
  첫 번째 호출에서 pool 생성 비용이 여전히 포함됨.

### 3-5. 두 접근법 비교

| 기준 | Parallel GEMM | Freivalds (GEMV) |
|------|--------------|-----------------|
| 시간복잡도 | O(N³) | O(N²) × 3회 |
| 정확도 | Deterministic | 확률적 (오류율 ≤ 12.5%) |
| 측정 시간 | ~0.488s | ~0.0016-0.002s |
| 적합한 용도 | 결과값 자체가 필요할 때 | 결과의 정합성만 검증할 때 |
| 병렬화 이득 | 스레드 수에 선형 비례 | Thread 생성 비용에 의해 제한 |

**결론**: 단순 검증 목적이라면 Freivalds가 압도적으로 빠름 (~300x). 단, 결과의 정확도가 필요하면 GEMM이 필수.

---

## 실험 결과 기록란 (측정 후 채울 것)

```
[A] Thread Creation Cost Only
  1 thread:  ___ ms (___ us/thread)
  4 threads: ___ ms (___ us/thread)
  8 threads: ___ ms (___ us/thread)
 16 threads: ___ ms (___ us/thread)
 32 threads: ___ ms (___ us/thread)
 64 threads: ___ ms (___ us/thread)

[B] Per-call thread (Freivalds x3): ___ ms

[C] Pool inside timer:              ___ ms
    (pool creation portion):        ___ ms

[D] Pool outside timer (compute):   ___ ms
    Pool creation time:             ___ ms

[E] Per-gemv breakdown (pool outside)
    gemv #1: ___ ms
    gemv #2: ___ ms
    gemv #3: ___ ms

핵심 비교:
  C - D (pool 생성 비용)          = ___ ms
  B - D (per-call 생성 overhead)  = ___ ms
  (B-D)/3 ≈ A의 15thread 측정값  = ___ ms  [일치하면 가설 증명]
```
