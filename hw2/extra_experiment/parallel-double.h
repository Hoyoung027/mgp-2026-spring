// Architecture detection: AVX2+FMA only on x86
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#define HAVE_AVX2
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>
// You cannot use OpenMP <omp.h>

// CPU spin hint (reduces power + hyper-threading contention during busy-wait)
static inline void spin_pause() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// ============================================================
// Thread Pool: spin-wait 기반, thread 재생성 오버헤드 제거
//   - Freivalds에서 gemv() 3회 호출 시 thread 생성 비용 제거
//   - 각 worker는 atomic phase를 spin-wait하며 work dispatch를 대기
// ============================================================
struct ThreadPool {
    // 64-byte alignment: false sharing 방지 (다른 worker의 phase가 같은 캐시라인에 놓이지 않게)
    struct alignas(64) State { std::atomic<int> phase{0}; };

    int T;
    std::vector<std::thread> workers;
    std::function<void(int, int)> task;  // task(tid, total_threads)
    std::vector<State> states;
    std::atomic<int> done{0};
    std::atomic<bool> alive{true};
    int cur_phase{0};

    explicit ThreadPool(int t) : T(t), states(t) {
        workers.reserve(t);
        for (int i = 0; i < t; i++) {
            workers.emplace_back([this, i]() {
                int last = 0;
                while (true) {
                    // 새 phase가 올 때까지 spin-wait
                    while (states[i].phase.load(std::memory_order_acquire) == last)
                        spin_pause();
                    last = states[i].phase.load(std::memory_order_relaxed);
                    if (!alive.load(std::memory_order_relaxed)) return;
                    task(i, T);
                    done.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }

    void run(std::function<void(int, int)> fn) {
        task = std::move(fn);
        done.store(0, std::memory_order_relaxed);
        ++cur_phase;
        // 모든 worker에게 새 phase 알림 (release: task 쓰기가 worker에게 보임을 보장)
        for (int i = 0; i < T; i++)
            states[i].phase.store(cur_phase, std::memory_order_release);
        // 모든 worker 완료까지 spin-wait (acquire: worker의 쓰기 결과가 main에 보임을 보장)
        while (done.load(std::memory_order_acquire) < T)
            spin_pause();
    }

    ~ThreadPool() {
        alive.store(false, std::memory_order_relaxed);
        ++cur_phase;
        for (int i = 0; i < T; i++)
            states[i].phase.store(cur_phase, std::memory_order_release);
        for (auto& w : workers) w.join();
    }
};

// 프로그램 수명 동안 단 1회 생성 (C++11 static local: thread-safe initialization)
inline ThreadPool& get_pool() {
    static ThreadPool pool(std::min((int)std::thread::hardware_concurrency(), 32));
    return pool;
}

// ============================================================
// AVX2 helper: __m256d 4개 double의 수평 합산
// ============================================================
#ifdef HAVE_AVX2
static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s2 = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
}
#endif

// ============================================================
// init_vec: Freivalds용 랜덤 벡터 초기화
// ============================================================
inline void init_vec(double* a, int N) {
    srand(42);
    for (int i = 0; i < N; ++i)
        a[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// ============================================================
// GEMV: 행-병렬 + AVX2 FMA + 2행 동시 계산
//
// 최적화 원리:
//   1. Thread pool: 3회 반복 호출에서 thread 재생성 오버헤드 제거
//   2. AVX2 FMA: 4 doubles/instruction (scalar 대비 4× throughput)
//   3. 2행 동시: b[j] 1회 로드로 2행에 재사용 → b[] 메모리 접근 절반
//   4. 4× unroll: 16 doubles/iteration, ILP 최대화
// ============================================================
inline void gemv(double* a, double* b, double* c, int N) {
    ThreadPool& pool = get_pool();
    const int T     = pool.T;
    const int chunk = N / T;

    pool.run([=](int tid, int P) {
        const int start = tid * chunk;
        const int end   = (tid == P - 1) ? N : start + chunk;

#ifdef HAVE_AVX2
        // ── AVX2 경로 (x86 채점 서버) ──────────────────────────────
        int i = start;
        // 2행 동시: b[] 로드를 두 행에 재사용
        for (; i <= end - 2; i += 2) {
            const double* r0 = a + i * N;
            const double* r1 = r0 + N;
            // 4개 누산기 × 2행 = 8개 AVX2 레지스터: 독립 의존성으로 ILP 극대화
            __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
            __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
            __m256d t0 = _mm256_setzero_pd(), t1 = _mm256_setzero_pd();
            __m256d t2 = _mm256_setzero_pd(), t3 = _mm256_setzero_pd();

            int j = 0;
            for (; j <= N - 16; j += 16) {
                // b[j..j+15]: 4개 AVX2 레지스터에 로드 → 양 행에 재사용
                __m256d bv0 = _mm256_loadu_pd(b + j);
                __m256d bv1 = _mm256_loadu_pd(b + j + 4);
                __m256d bv2 = _mm256_loadu_pd(b + j + 8);
                __m256d bv3 = _mm256_loadu_pd(b + j + 12);
                // row0: fused multiply-add (1 instruction = mul + add)
                s0 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j),      bv0, s0);
                s1 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 4),  bv1, s1);
                s2 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 8),  bv2, s2);
                s3 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 12), bv3, s3);
                // row1: bv0~bv3 재사용 (추가 메모리 접근 없음)
                t0 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j),      bv0, t0);
                t1 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 4),  bv1, t1);
                t2 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 8),  bv2, t2);
                t3 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 12), bv3, t3);
            }
            // 4개 누산기 합산 후 수평 합산
            c[i]     = hsum256(_mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
            c[i + 1] = hsum256(_mm256_add_pd(_mm256_add_pd(t0, t1), _mm256_add_pd(t2, t3)));
            // remainder (N=2048은 16의 배수이므로 실질적으로 실행 안 됨)
            for (; j < N; j++) { c[i] += r0[j] * b[j]; c[i + 1] += r1[j] * b[j]; }
        }
        // 홀수 행 처리 (end-start가 홀수인 경우)
        for (; i < end; i++) {
            const double* r = a + i * N;
            __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
            __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
            int j = 0;
            for (; j <= N - 16; j += 16) {
                s0 = _mm256_fmadd_pd(_mm256_loadu_pd(r + j),      _mm256_loadu_pd(b + j),      s0);
                s1 = _mm256_fmadd_pd(_mm256_loadu_pd(r + j + 4),  _mm256_loadu_pd(b + j + 4),  s1);
                s2 = _mm256_fmadd_pd(_mm256_loadu_pd(r + j + 8),  _mm256_loadu_pd(b + j + 8),  s2);
                s3 = _mm256_fmadd_pd(_mm256_loadu_pd(r + j + 12), _mm256_loadu_pd(b + j + 12), s3);
            }
            c[i] = hsum256(_mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
            for (; j < N; j++) c[i] += r[j] * b[j];
        }

#else
        // ── Scalar fallback (ARM 로컬 빌드) ────────────────────────
        // 2행 동시 + 4-way scalar unroll + 포인터 산술
        const double* b_end = b + N - 3;  // 4-way 루프 종료 조건 사전 계산
        double* row0 = a + start * N;
        double* row1 = row0 + N;
        int i = start;
        for (; i <= end - 2; i += 2, row0 += 2 * N, row1 += 2 * N) {
            double s0=0, s1=0, s2=0, s3=0, t0=0, t1=0, t2=0, t3=0;
            const double* pb  = b;
            const double* pr0 = row0;
            const double* pr1 = row1;
            for (; pb < b_end; pb+=4, pr0+=4, pr1+=4) {
                double bj0=pb[0], bj1=pb[1], bj2=pb[2], bj3=pb[3];
                s0+=pr0[0]*bj0; t0+=pr1[0]*bj0;
                s1+=pr0[1]*bj1; t1+=pr1[1]*bj1;
                s2+=pr0[2]*bj2; t2+=pr1[2]*bj2;
                s3+=pr0[3]*bj3; t3+=pr1[3]*bj3;
            }
            c[i]     = s0+s1+s2+s3;
            c[i + 1] = t0+t1+t2+t3;
        }
        for (; i < end; i++, row0 += N) {
            double s0=0, s1=0, s2=0, s3=0;
            const double* pb  = b;
            const double* pr0 = row0;
            for (; pb < b_end; pb+=4, pr0+=4) {
                s0+=pr0[0]*pb[0]; s1+=pr0[1]*pb[1];
                s2+=pr0[2]*pb[2]; s3+=pr0[3]*pb[3];
            }
            c[i] = s0+s1+s2+s3;
        }
#endif
    });
}

// ============================================================
// GEMM: Cache Tiling + AVX2 FMA micro-kernel
//
// 최적화 원리:
//   1. Cache Tiling (BS=64): A/B/C 타일을 L1 캐시에 유지
//      → k 루프가 진행되면서 B 행이 캐시에서 밀려나는 문제 해결
//      → N=2048 기준 BS=64: 타일 1개 = 32KB = L1 딱 맞음
//   2. AVX2 FMA micro-kernel: 내부 j 루프에서 16 doubles/iter
//   3. Thread pool: i-tile 단위 분배, 행이 겹치지 않아 동기화 불필요
// ============================================================
inline void gemm(double* a, double* b, double* c, int N) {
    const int BS = 64;  // tile size (64×64×8B = 32KB)
    ThreadPool& pool = get_pool();
    const int T = pool.T;

    // C 행렬 0으로 초기화 (병렬)
    pool.run([=](int tid, int P) {
        int chunk = (N + P - 1) / P;
        int s = tid * chunk;
        int e = std::min(s + chunk, N);
        for (int i = s; i < e; i++)
            for (int j = 0; j < N; j++) c[i * N + j] = 0.0;
    });

    // i-tile을 thread 간 분배
    const int n_ti = (N + BS - 1) / BS;  // = 32 (N=2048, BS=64)
    const int n_tk = (N + BS - 1) / BS;
    const int n_tj = (N + BS - 1) / BS;

    pool.run([=](int tid, int P) {
        // 각 thread가 담당할 i-tile 범위
        int tiles_per = (n_ti + P - 1) / P;
        int ti_s = tid * tiles_per;
        int ti_e = std::min(ti_s + tiles_per, n_ti);

        for (int ti = ti_s; ti < ti_e; ti++) {
            int i_s = ti * BS, i_e = std::min(i_s + BS, N);

            for (int tk = 0; tk < n_tk; tk++) {
                int k_s = tk * BS, k_e = std::min(k_s + BS, N);

                for (int tj = 0; tj < n_tj; tj++) {
                    int j_s = tj * BS, j_e = std::min(j_s + BS, N);

                    // Micro-kernel: 타일 (ti, tk, tj) 계산
                    for (int i = i_s; i < i_e; i++) {
                        for (int k = k_s; k < k_e; k++) {
                            double aik = a[i * N + k];
#ifdef HAVE_AVX2
                            // aik를 256비트 레지스터 4개 레인에 broadcast
                            __m256d va = _mm256_set1_pd(aik);
                            int j = j_s;
                            // 16 doubles/iter: c[i][j..j+15] += aik * b[k][j..j+15]
                            for (; j <= j_e - 16; j += 16) {
                                _mm256_storeu_pd(c+i*N+j,
                                    _mm256_fmadd_pd(va, _mm256_loadu_pd(b+k*N+j),
                                                        _mm256_loadu_pd(c+i*N+j)));
                                _mm256_storeu_pd(c+i*N+j+4,
                                    _mm256_fmadd_pd(va, _mm256_loadu_pd(b+k*N+j+4),
                                                        _mm256_loadu_pd(c+i*N+j+4)));
                                _mm256_storeu_pd(c+i*N+j+8,
                                    _mm256_fmadd_pd(va, _mm256_loadu_pd(b+k*N+j+8),
                                                        _mm256_loadu_pd(c+i*N+j+8)));
                                _mm256_storeu_pd(c+i*N+j+12,
                                    _mm256_fmadd_pd(va, _mm256_loadu_pd(b+k*N+j+12),
                                                        _mm256_loadu_pd(c+i*N+j+12)));
                            }
                            for (; j < j_e; j++) c[i*N+j] += aik * b[k*N+j];
#else
                            // Scalar fallback: 4-way unroll
                            int j = j_s;
                            for (; j <= j_e - 4; j += 4) {
                                c[i*N+j]   += aik * b[k*N+j];
                                c[i*N+j+1] += aik * b[k*N+j+1];
                                c[i*N+j+2] += aik * b[k*N+j+2];
                                c[i*N+j+3] += aik * b[k*N+j+3];
                            }
                            for (; j < j_e; j++) c[i*N+j] += aik * b[k*N+j];
#endif
                        }
                    }
                }
            }
        }
    });
}
