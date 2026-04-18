#include <immintrin.h>

#include <algorithm>
#include <thread>
#include <vector>

// ============================================================
// gemv: naive
// ============================================================
inline void gemv_naive(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            for (int i = start; i < end; i++) {
                double s = 0.0;
                for (int j = 0; j < N; j++)
                    s += a[i * N + j] * b[j];
                c[i] = s;
            }
        });
    }
    for (auto &th : threads) th.join();
}

// ============================================================
// gemv: unrolled (scalar 4-unroll + 2행 동시 처리)
// ============================================================
inline void gemv_unrolled(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            const double *b_end = b + N - 3;
            double *row0 = a + start * N;
            double *row1 = row0 + N;
            int i = start;
            for (; i <= end - 2; i += 2, row0 += 2*N, row1 += 2*N) {
                double s0=0, s1=0, s2=0, s3=0;
                double t0=0, t1=0, t2=0, t3=0;
                const double *pb  = b;
                const double *pr0 = row0;
                const double *pr1 = row1;
                for (; pb < b_end; pb+=4, pr0+=4, pr1+=4) {
                    double bj0=pb[0], bj1=pb[1], bj2=pb[2], bj3=pb[3];
                    s0 += pr0[0]*bj0;  t0 += pr1[0]*bj0;
                    s1 += pr0[1]*bj1;  t1 += pr1[1]*bj1;
                    s2 += pr0[2]*bj2;  t2 += pr1[2]*bj2;
                    s3 += pr0[3]*bj3;  t3 += pr1[3]*bj3;
                }
                double acc0 = s0+s1+s2+s3, acc1 = t0+t1+t2+t3;
                for (; pb < b+N; pb++, pr0++, pr1++) {
                    acc0 += *pr0 * *pb;
                    acc1 += *pr1 * *pb;
                }
                c[i]     = acc0;
                c[i + 1] = acc1;
            }
            for (; i < end; i++, row0 += N) {
                double s0=0, s1=0, s2=0, s3=0;
                const double *pb  = b;
                const double *pr0 = row0;
                for (; pb < b_end; pb+=4, pr0+=4) {
                    s0 += pr0[0]*pb[0];
                    s1 += pr0[1]*pb[1];
                    s2 += pr0[2]*pb[2];
                    s3 += pr0[3]*pb[3];
                }
                double acc = s0+s1+s2+s3;
                for (; pb < b+N; pb++, pr0++) acc += *pr0 * *pb;
                c[i] = acc;
            }
        });
    }
    for (auto &th : threads) th.join();
}

// ============================================================
// gemv: SSE (__m128d 8-unroll + 2행 동시 처리)
// ============================================================
inline void gemv_sse(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            const double *b_end8 = b + N - 7;
            const double *b_end  = b + N;
            double *row0 = a + start * N;
            double *row1 = row0 + N;
            int i = start;
            for (; i <= end - 2; i += 2, row0 += 2*N, row1 += 2*N) {
                __m128d s0=_mm_setzero_pd(), s1=_mm_setzero_pd();
                __m128d s2=_mm_setzero_pd(), s3=_mm_setzero_pd();
                __m128d t0=_mm_setzero_pd(), t1=_mm_setzero_pd();
                __m128d t2=_mm_setzero_pd(), t3=_mm_setzero_pd();
                const double *pb  = b;
                const double *pr0 = row0;
                const double *pr1 = row1;
                for (; pb < b_end8; pb+=8, pr0+=8, pr1+=8) {
                    __m128d bv0=_mm_loadu_pd(pb+0), bv1=_mm_loadu_pd(pb+2);
                    __m128d bv2=_mm_loadu_pd(pb+4), bv3=_mm_loadu_pd(pb+6);
                    s0=_mm_add_pd(s0, _mm_mul_pd(_mm_loadu_pd(pr0+0), bv0));
                    s1=_mm_add_pd(s1, _mm_mul_pd(_mm_loadu_pd(pr0+2), bv1));
                    s2=_mm_add_pd(s2, _mm_mul_pd(_mm_loadu_pd(pr0+4), bv2));
                    s3=_mm_add_pd(s3, _mm_mul_pd(_mm_loadu_pd(pr0+6), bv3));
                    t0=_mm_add_pd(t0, _mm_mul_pd(_mm_loadu_pd(pr1+0), bv0));
                    t1=_mm_add_pd(t1, _mm_mul_pd(_mm_loadu_pd(pr1+2), bv1));
                    t2=_mm_add_pd(t2, _mm_mul_pd(_mm_loadu_pd(pr1+4), bv2));
                    t3=_mm_add_pd(t3, _mm_mul_pd(_mm_loadu_pd(pr1+6), bv3));
                }
                __m128d rs = _mm_add_pd(_mm_add_pd(s0, s1), _mm_add_pd(s2, s3));
                __m128d rt = _mm_add_pd(_mm_add_pd(t0, t1), _mm_add_pd(t2, t3));
                double acc0 = _mm_cvtsd_f64(_mm_add_sd(rs, _mm_unpackhi_pd(rs, rs)));
                double acc1 = _mm_cvtsd_f64(_mm_add_sd(rt, _mm_unpackhi_pd(rt, rt)));
                for (; pb < b_end; pb++, pr0++, pr1++) {
                    acc0 += *pr0 * *pb;
                    acc1 += *pr1 * *pb;
                }
                c[i]     = acc0;
                c[i + 1] = acc1;
            }
            for (; i < end; i++, row0 += N) {
                __m128d s0=_mm_setzero_pd(), s1=_mm_setzero_pd();
                __m128d s2=_mm_setzero_pd(), s3=_mm_setzero_pd();
                const double *pb  = b;
                const double *pr0 = row0;
                for (; pb < b_end8; pb+=8, pr0+=8) {
                    s0=_mm_add_pd(s0, _mm_mul_pd(_mm_loadu_pd(pr0+0), _mm_loadu_pd(pb+0)));
                    s1=_mm_add_pd(s1, _mm_mul_pd(_mm_loadu_pd(pr0+2), _mm_loadu_pd(pb+2)));
                    s2=_mm_add_pd(s2, _mm_mul_pd(_mm_loadu_pd(pr0+4), _mm_loadu_pd(pb+4)));
                    s3=_mm_add_pd(s3, _mm_mul_pd(_mm_loadu_pd(pr0+6), _mm_loadu_pd(pb+6)));
                }
                __m128d rs = _mm_add_pd(_mm_add_pd(s0, s1), _mm_add_pd(s2, s3));
                double acc = _mm_cvtsd_f64(_mm_add_sd(rs, _mm_unpackhi_pd(rs, rs)));
                for (; pb < b_end; pb++, pr0++) acc += *pr0 * *pb;
                c[i] = acc;
            }
        });
    }
    for (auto &th : threads) th.join();
}

// ============================================================
// gemm: naive (i-j-k)
// ============================================================
inline void gemm_naive(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            for (int i = start; i < end; i++)
                for (int j = 0; j < N; j++) c[i*N+j] = 0.0;
            for (int i = start; i < end; i++)
                for (int j = 0; j < N; j++)
                    for (int k = 0; k < N; k++)
                        c[i*N+j] += a[i*N+k] * b[k*N+j];
        });
    }
    for (auto &th : threads) th.join();
}

// ============================================================
// gemm: unrolled (i-k-j, scalar 4-unroll)
// ============================================================
inline void gemm_unrolled(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            for (int i = start; i < end; i++)
                for (int j = 0; j < N; j++) c[i*N+j] = 0.0;
            for (int i = start; i < end; i++)
                for (int k = 0; k < N; k++) {
                    double aik = a[i*N+k];
                    int j = 0;
                    for (; j <= N - 4; j += 4) {
                        c[i*N+j]   += aik * b[k*N+j];
                        c[i*N+j+1] += aik * b[k*N+j+1];
                        c[i*N+j+2] += aik * b[k*N+j+2];
                        c[i*N+j+3] += aik * b[k*N+j+3];
                    }
                    for (; j < N; j++) c[i*N+j] += aik * b[k*N+j];
                }
        });
    }
    for (auto &th : threads) th.join();
}

// ============================================================
// gemm: SSE (i-k-j, __m128d broadcast)
// ============================================================
inline void gemm_sse(double *a, double *b, double *c, int N, int T) {
    std::vector<std::thread> threads;
    threads.reserve(T);
    int chunk = N / T;
    for (int t = 0; t < T; t++) {
        int start = t * chunk;
        int end   = (t == T - 1) ? N : start + chunk;
        threads.emplace_back([=]() {
            for (int i = start; i < end; i++)
                for (int j = 0; j < N; j++) c[i*N+j] = 0.0;
            for (int i = start; i < end; i++)
                for (int k = 0; k < N; k++) {
                    __m128d aik_v = _mm_set1_pd(a[i*N+k]);
                    int j = 0;
                    for (; j <= N - 2; j += 2) {
                        _mm_storeu_pd(c + i*N + j,
                            _mm_add_pd(_mm_loadu_pd(c + i*N + j),
                                _mm_mul_pd(aik_v, _mm_loadu_pd(b + k*N + j))));
                    }
                    for (; j < N; j++) c[i*N+j] += a[i*N+k] * b[k*N+j];
                }
        });
    }
    for (auto &th : threads) th.join();
}
