#pragma GCC target("avx2,fma")
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
// You cannot use OpenMP <omp.h>

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s2 = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
}

inline void init_vec(double* a, int N) {
    srand(42);
    for (int i = 0; i < N; ++i)
        a[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// GEMV: 행-병렬 + AVX2 FMA + 2행 동시 계산
//   1. AVX2 FMA: 4 doubles/instruction (scalar 대비 4× throughput)
//   2. 2행 동시: b[j] 1회 로드로 2행에 재사용 → b[] 메모리 접근 절반
//   3. 4× unroll: 16 doubles/iteration, ILP 최대화
inline void gemv(double* a, double* b, double* c, int N) {
    const int T     = std::max(1, std::min((int)std::thread::hardware_concurrency(), 8));
    const int chunk = N / T;
    std::vector<std::thread> threads;
    threads.reserve(T);

    for (int t = 0; t < T; t++) {
        const int start = t * chunk;
        const int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            int i = start;
            for (; i <= end - 2; i += 2) {
                const double* r0 = a + i * N;
                const double* r1 = r0 + N;
                __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
                __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
                __m256d t0 = _mm256_setzero_pd(), t1 = _mm256_setzero_pd();
                __m256d t2 = _mm256_setzero_pd(), t3 = _mm256_setzero_pd();

                int j = 0;
                for (; j <= N - 16; j += 16) {
                    __m256d bv0 = _mm256_loadu_pd(b + j);
                    __m256d bv1 = _mm256_loadu_pd(b + j + 4);
                    __m256d bv2 = _mm256_loadu_pd(b + j + 8);
                    __m256d bv3 = _mm256_loadu_pd(b + j + 12);
                    s0 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j),      bv0, s0);
                    s1 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 4),  bv1, s1);
                    s2 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 8),  bv2, s2);
                    s3 = _mm256_fmadd_pd(_mm256_loadu_pd(r0 + j + 12), bv3, s3);
                    t0 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j),      bv0, t0);
                    t1 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 4),  bv1, t1);
                    t2 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 8),  bv2, t2);
                    t3 = _mm256_fmadd_pd(_mm256_loadu_pd(r1 + j + 12), bv3, t3);
                }
                c[i]     = hsum256(_mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
                c[i + 1] = hsum256(_mm256_add_pd(_mm256_add_pd(t0, t1), _mm256_add_pd(t2, t3)));
                for (; j < N; j++) { c[i] += r0[j] * b[j]; c[i + 1] += r1[j] * b[j]; }
            }
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
        });
    }
    for (auto& th : threads) th.join();
}

// GEMM: Cache Tiling (BS=64) + AVX2 FMA micro-kernel
//   1. Cache Tiling: A/B/C 타일을 L1에 유지 (64×64×8B = 32KB)
//   2. AVX2 FMA: 16 doubles/iter
//   3. i-tile 단위 thread 분배: 행 겹침 없음, 동기화 불필요
inline void gemm(double* a, double* b, double* c, int N) {
    const int BS   = 64;
    const int T    = std::max(1, std::min((int)std::thread::hardware_concurrency(), 32));
    const int n_ti = (N + BS - 1) / BS;
    const int n_tk = (N + BS - 1) / BS;
    const int n_tj = (N + BS - 1) / BS;

    std::vector<std::thread> threads;
    threads.reserve(T);

    for (int t = 0; t < T; t++) {
        threads.emplace_back([=]() {
            int tiles_per = (n_ti + T - 1) / T;
            int ti_s = t * tiles_per;
            int ti_e = std::min(ti_s + tiles_per, n_ti);

            int row_s = ti_s * BS, row_e = std::min(ti_e * BS, N);
            for (int i = row_s; i < row_e; i++)
                for (int j = 0; j < N; j++) c[i * N + j] = 0.0;

            for (int ti = ti_s; ti < ti_e; ti++) {
                int i_s = ti * BS, i_e = std::min(i_s + BS, N);
                for (int tk = 0; tk < n_tk; tk++) {
                    int k_s = tk * BS, k_e = std::min(k_s + BS, N);
                    for (int tj = 0; tj < n_tj; tj++) {
                        int j_s = tj * BS, j_e = std::min(j_s + BS, N);
                        for (int i = i_s; i < i_e; i++) {
                            for (int k = k_s; k < k_e; k++) {
                                __m256d va = _mm256_set1_pd(a[i * N + k]);
                                int j = j_s;
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
                                for (; j < j_e; j++) c[i*N+j] += a[i*N+k] * b[k*N+j];
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}
