#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <numeric>
#include <thread>
#include <vector>

// ============================================================
// GemvPool: 재사용 가능한 스레드 풀 (외부에서 생성 가능)
// ============================================================
struct GemvPool {
    int nt;
    std::vector<std::thread> workers;
    std::atomic<float*> g_a{nullptr}, g_b{nullptr}, g_c{nullptr};
    std::atomic<int>    g_N{0};
    std::atomic<int>    work_ver{0};
    std::atomic<int>    done_cnt{0};
    std::atomic<bool>   quit{false};

    void compute(float *a, float *b, float *c, int N, int i0, int i1) {
        auto hsum = [](__m128 v) {
            __m128 t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,0,3,2));
            v = _mm_add_ps(v, t);
            t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
            return _mm_cvtss_f32(_mm_add_ps(v, t));
        };
        int i = i0;
        for (; i + 3 < i1; i += 4) {
            const float *ai0=a+i*N, *ai1=a+(i+1)*N;
            const float *ai2=a+(i+2)*N, *ai3=a+(i+3)*N;
            __m128 vs0a=_mm_setzero_ps(), vs0b=_mm_setzero_ps();
            __m128 vs1a=_mm_setzero_ps(), vs1b=_mm_setzero_ps();
            __m128 vs2a=_mm_setzero_ps(), vs2b=_mm_setzero_ps();
            __m128 vs3a=_mm_setzero_ps(), vs3b=_mm_setzero_ps();
            int k = 0;
            for (; k <= N-8; k += 8) {
                __m128 vb0 = _mm_loadu_ps(b+k);
                __m128 vb1 = _mm_loadu_ps(b+k+4);
                vs0a = _mm_add_ps(vs0a, _mm_mul_ps(_mm_loadu_ps(ai0+k),   vb0));
                vs0b = _mm_add_ps(vs0b, _mm_mul_ps(_mm_loadu_ps(ai0+k+4), vb1));
                vs1a = _mm_add_ps(vs1a, _mm_mul_ps(_mm_loadu_ps(ai1+k),   vb0));
                vs1b = _mm_add_ps(vs1b, _mm_mul_ps(_mm_loadu_ps(ai1+k+4), vb1));
                vs2a = _mm_add_ps(vs2a, _mm_mul_ps(_mm_loadu_ps(ai2+k),   vb0));
                vs2b = _mm_add_ps(vs2b, _mm_mul_ps(_mm_loadu_ps(ai2+k+4), vb1));
                vs3a = _mm_add_ps(vs3a, _mm_mul_ps(_mm_loadu_ps(ai3+k),   vb0));
                vs3b = _mm_add_ps(vs3b, _mm_mul_ps(_mm_loadu_ps(ai3+k+4), vb1));
            }
            float s0=hsum(_mm_add_ps(vs0a,vs0b)), s1=hsum(_mm_add_ps(vs1a,vs1b));
            float s2=hsum(_mm_add_ps(vs2a,vs2b)), s3=hsum(_mm_add_ps(vs3a,vs3b));
            for (; k < N; k++) {
                float bk=b[k];
                s0+=ai0[k]*bk; s1+=ai1[k]*bk; s2+=ai2[k]*bk; s3+=ai3[k]*bk;
            }
            c[i]=s0; c[i+1]=s1; c[i+2]=s2; c[i+3]=s3;
        }
        for (; i < i1; i++) {
            const float *ai = a+i*N;
            __m128 vs0=_mm_setzero_ps(), vs1=_mm_setzero_ps();
            int k = 0;
            for (; k <= N-8; k += 8) {
                vs0 = _mm_add_ps(vs0, _mm_mul_ps(_mm_loadu_ps(ai+k),   _mm_loadu_ps(b+k)));
                vs1 = _mm_add_ps(vs1, _mm_mul_ps(_mm_loadu_ps(ai+k+4), _mm_loadu_ps(b+k+4)));
            }
            float sum = hsum(_mm_add_ps(vs0, vs1));
            for (; k < N; k++) sum += ai[k]*b[k];
            c[i] = sum;
        }
    }

    explicit GemvPool(int n) : nt(n) {
        workers.reserve(n-1);
        for (int tid = 1; tid < n; ++tid) {
            workers.emplace_back([this, tid]() {
                int seen = 0;
                while (true) {
                    int v, spin = 0;
                    while ((v = work_ver.load(std::memory_order_acquire)) == seen) {
                        if (++spin < 8000) _mm_pause();
                        else std::this_thread::yield();
                    }
                    seen = v;
                    if (quit.load(std::memory_order_relaxed)) return;
                    float *a=g_a.load(std::memory_order_relaxed);
                    float *b=g_b.load(std::memory_order_relaxed);
                    float *c=g_c.load(std::memory_order_relaxed);
                    int    N=g_N.load(std::memory_order_relaxed);
                    int rows_per = N / nt;
                    int i0 = tid * rows_per;
                    int i1 = (tid == nt-1) ? N : i0 + rows_per;
                    compute(a, b, c, N, i0, i1);
                    done_cnt.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }

    ~GemvPool() {
        quit.store(true, std::memory_order_relaxed);
        work_ver.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void run(float *a, float *b, float *c, int N) {
        g_a.store(a, std::memory_order_relaxed);
        g_b.store(b, std::memory_order_relaxed);
        g_c.store(c, std::memory_order_relaxed);
        g_N.store(N, std::memory_order_relaxed);
        done_cnt.store(0, std::memory_order_relaxed);
        work_ver.fetch_add(1, std::memory_order_release);
        int rows_per = N / nt;
        compute(a, b, c, N, 0, rows_per);
        int spin = 0;
        while (done_cnt.load(std::memory_order_acquire) < nt-1) {
            if (++spin < 2000) _mm_pause();
            else std::this_thread::yield();
        }
    }
};

// 외부에서 주입할 전역 풀 포인터
GemvPool* g_pool = nullptr;

// ============================================================
// gemv: g_pool이 설정되어 있으면 재사용, 아니면 per-call thread
// ============================================================
inline void gemv(float *a, float *b, float *c, int N) {
    if (g_pool) {
        g_pool->run(a, b, c, N);
        return;
    }
    // fallback: per-call thread (비교 baseline)
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([=]() {
            int rows_per = N / num_threads;
            int i_start  = tid * rows_per;
            int i_end    = (tid == num_threads-1) ? N : i_start + rows_per;
            for (int i = i_start; i < i_end; ++i) {
                const float *ai = a + i*N;
                __m128 vs0=_mm_setzero_ps(), vs1=_mm_setzero_ps();
                int k = 0;
                for (; k <= N-8; k += 8) {
                    vs0 = _mm_add_ps(vs0, _mm_mul_ps(_mm_loadu_ps(ai+k),   _mm_loadu_ps(b+k)));
                    vs1 = _mm_add_ps(vs1, _mm_mul_ps(_mm_loadu_ps(ai+k+4), _mm_loadu_ps(b+k+4)));
                }
                __m128 v = _mm_add_ps(vs0, vs1);
                __m128 t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,0,3,2));
                v = _mm_add_ps(v, t); t = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
                float sum = _mm_cvtss_f32(_mm_add_ps(v, t));
                for (; k < N; k++) sum += ai[k]*b[k];
                c[i] = sum;
            }
        });
    }
    for (auto &t : threads) t.join();
}

inline void init_vec(float *a, int N) {
    srand(42);
    for (int i = 0; i < N; ++i)
        a[i] = (float)rand() / (float)RAND_MAX;
}

inline void gemm(float *a, float *b, float *c, int N) {
    const int TILE = 64;
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 64) num_threads = 64;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([=]() {
            int rows_per = N / num_threads;
            int i_start  = tid * rows_per;
            int i_end    = (tid == num_threads-1) ? N : i_start + rows_per;
            for (int i = i_start; i < i_end; ++i)
                for (int j = 0; j < N; ++j)
                    c[i*N+j] = 0.0f;
            for (int ii = i_start; ii < i_end; ii += TILE) {
                int i_lim = std::min(ii+TILE, i_end);
                for (int kk = 0; kk < N; kk += TILE) {
                    int k_lim = std::min(kk+TILE, N);
                    for (int jj = 0; jj < N; jj += TILE) {
                        int j_lim = std::min(jj+TILE, N);
                        for (int i = ii; i < i_lim; ++i) {
                            for (int k = kk; k < k_lim; ++k) {
                                __m128 va = _mm_set1_ps(a[i*N+k]);
                                float *cp = c+i*N;
                                const float *bp = b+k*N;
                                int j = jj;
                                for (; j <= j_lim-16; j += 16) {
                                    _mm_storeu_ps(cp+j,    _mm_add_ps(_mm_loadu_ps(cp+j),    _mm_mul_ps(va, _mm_loadu_ps(bp+j))));
                                    _mm_storeu_ps(cp+j+4,  _mm_add_ps(_mm_loadu_ps(cp+j+4),  _mm_mul_ps(va, _mm_loadu_ps(bp+j+4))));
                                    _mm_storeu_ps(cp+j+8,  _mm_add_ps(_mm_loadu_ps(cp+j+8),  _mm_mul_ps(va, _mm_loadu_ps(bp+j+8))));
                                    _mm_storeu_ps(cp+j+12, _mm_add_ps(_mm_loadu_ps(cp+j+12), _mm_mul_ps(va, _mm_loadu_ps(bp+j+12))));
                                }
                                for (; j <= j_lim-4; j += 4)
                                    _mm_storeu_ps(cp+j, _mm_add_ps(_mm_loadu_ps(cp+j), _mm_mul_ps(va, _mm_loadu_ps(bp+j))));
                                for (; j < j_lim; ++j)
                                    cp[j] += a[i*N+k] * bp[j];
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto &t : threads) t.join();
}
