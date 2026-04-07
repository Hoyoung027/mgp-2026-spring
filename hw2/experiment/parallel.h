#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <numeric>
#include <thread>
#include <vector>
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile

/**
 * @brief Initializes a vector with random float values.
 *
 * This function fills the given array
 *
 * @param a Pointer to the array to be initialized.
 * @param N The number of elements in the array.
 */
inline void init_vec(float *a, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

	srand(42);
	for (int i = 0; i < N; ++i)
		a[i] = (float)rand() / (float)RAND_MAX;

	/****************/
}

/**
 * @brief Performs a matrix-vector multiplication.
 *
 * This function computes the product of a matrix 'a' and a vector 'b', storing
 * the result in vector 'c'.
 *
 * @param a Pointer to the first element of the matrix 'a' (assumed to be in
 * row-major order).
 * @param b Pointer to the first element of the vector 'b'.
 * @param c Pointer to the first element of the result vector 'c'.
 * @param N The dimension of the matrix and vectors (assuming a square matrix
 * and compatible vector sizes).
 */
inline void gemv(float *a, float *b, float *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

	// ── Static thread pool (함수 내부 struct, TODO 영역 안에서 완결) ──────────
	struct GemvPool {
		int nt;
		std::vector<std::thread> workers;
		std::atomic<float*> g_a{nullptr}, g_b{nullptr}, g_c{nullptr};
		std::atomic<int>    g_N{0};
		std::atomic<int>    work_ver{0};  // run()마다 +1 → 워커가 감지
		std::atomic<int>    done_cnt{0};  // 워커 완료 시 +1
		std::atomic<bool>   quit{false};

		// SSE2 dot-product: 행 [i0, i1) 계산
		static void compute(float *a, float *b, float *c, int N, int i0, int i1) {
			for (int i = i0; i < i1; ++i) {
				const float *ai = a + i * N;
				__m128 vs0 = _mm_setzero_ps(), vs1 = _mm_setzero_ps();
				int k = 0;
				for (; k <= N - 8; k += 8) {
					vs0 = _mm_add_ps(vs0, _mm_mul_ps(_mm_loadu_ps(ai+k),   _mm_loadu_ps(b+k)));
					vs1 = _mm_add_ps(vs1, _mm_mul_ps(_mm_loadu_ps(ai+k+4), _mm_loadu_ps(b+k+4)));
				}
				__m128 vs = _mm_add_ps(vs0, vs1);
				__m128 t  = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(1,0,3,2));
				vs = _mm_add_ps(vs, t);
				t  = _mm_shuffle_ps(vs, vs, _MM_SHUFFLE(2,3,0,1));
				vs = _mm_add_ps(vs, t);
				float sum = _mm_cvtss_f32(vs);
				for (; k < N; ++k) sum += ai[k] * b[k];
				c[i] = sum;
			}
		}

		explicit GemvPool(int n) : nt(n) {
			workers.reserve(n - 1);
			for (int tid = 1; tid < n; ++tid) {
				workers.emplace_back([this, tid]() {
					int seen = 0;
					while (true) {
						// spinwait: work_ver가 바뀔 때까지 대기
						int v;
						while ((v = work_ver.load(std::memory_order_acquire)) == seen)
							std::this_thread::yield();
						seen = v;
						if (quit.load(std::memory_order_relaxed)) return;

						float *a = g_a.load(std::memory_order_relaxed);
						float *b = g_b.load(std::memory_order_relaxed);
						float *c = g_c.load(std::memory_order_relaxed);
						int    N = g_N.load(std::memory_order_relaxed);

						int rows_per = N / nt;
						int i0 = tid * rows_per;
						int i1 = (tid == nt - 1) ? N : i0 + rows_per;
						compute(a, b, c, N, i0, i1);

						done_cnt.fetch_add(1, std::memory_order_release);
					}
				});
			}
		}

		~GemvPool() {
			quit.store(true, std::memory_order_relaxed);
			work_ver.fetch_add(1, std::memory_order_release); // 워커 깨워서 quit 확인
			for (auto &t : workers) t.join();
		}

		void run(float *a, float *b, float *c, int N) {
			g_a.store(a, std::memory_order_relaxed);
			g_b.store(b, std::memory_order_relaxed);
			g_c.store(c, std::memory_order_relaxed);
			g_N.store(N, std::memory_order_relaxed);
			done_cnt.store(0, std::memory_order_relaxed);
			work_ver.fetch_add(1, std::memory_order_release); // 워커 깨우기

			// 메인 스레드(tid=0) 담당 행 직접 처리
			int rows_per = N / nt;
			compute(a, b, c, N, 0, rows_per);

			// 모든 워커 완료 대기
			while (done_cnt.load(std::memory_order_acquire) < nt - 1)
				std::this_thread::yield();
		}
	};

	static int nt = [] {
		int n = (int)std::thread::hardware_concurrency();
		if (n < 1) n = 1;
		if (n > 64) n = 64;
		return n;
	}();
	static GemvPool pool(nt); // 첫 호출에만 생성, 이후 재사용

	pool.run(a, b, c, N);

	/****************/
}

/**
 * @brief Performs matrix multiplication of two NxN matrices.
 *
 * This function computes the product of two square matrices `a` and `b`,
 * and stores the result in matrix `c`. All matrices are represented as
 * 1-dimensional arrays in row-major order.
 *
 * @param a Pointer to the first input matrix (NxN).
 * @param b Pointer to the second input matrix (NxN).
 * @param c Pointer to the output matrix (NxN) where the result will be stored.
 * @param N The dimension of the matrices (number of rows and columns).
 */
inline void gemm(float *a, float *b, float *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

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
			int i_end    = (tid == num_threads - 1) ? N : i_start + rows_per;

			// zero assigned rows of c
			for (int i = i_start; i < i_end; ++i)
				for (int j = 0; j < N; ++j)
					c[i * N + j] = 0.0f;

			// tiled ikj GEMM
			for (int ii = i_start; ii < i_end; ii += TILE) {
				int i_lim = std::min(ii + TILE, i_end);
				for (int kk = 0; kk < N; kk += TILE) {
					int k_lim = std::min(kk + TILE, N);
					for (int jj = 0; jj < N; jj += TILE) {
						int j_lim = std::min(jj + TILE, N);
						for (int i = ii; i < i_lim; ++i) {
							for (int k = kk; k < k_lim; ++k) {
								float a_ik = a[i * N + k];
								for (int j = jj; j < j_lim; ++j)
									c[i * N + j] += a_ik * b[k * N + j];
							}
						}
					}
				}
			}
		});
	}

	for (auto &t : threads) t.join();

	/****************/
}
