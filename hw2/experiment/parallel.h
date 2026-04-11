#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

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

	// Task descriptor: each worker thread handles rows [i0, i1)
	struct Task {
		float *a, *b, *c;
		int i0, i1, N;
	};

	// Spin-wait thread pool (function-local static: created once, reused per call)
	struct Pool {
		int nt;
		std::vector<Task> tasks;
		std::vector<std::atomic<int>> state; // 0=idle, 1=run, -1=stop
		std::vector<std::thread> workers;

		static void run_task(const Task &t) {
			for (int i = t.i0; i < t.i1; ++i) {
				const float *ai = t.a + (size_t)i * t.N;
				float sum = 0.0f;
				for (int k = 0; k < t.N; ++k)
					sum += ai[k] * t.b[k];
				t.c[i] = sum;
			}
		}

		Pool(int n) : nt(n), tasks(n), state(n) {
			for (int i = 0; i < n; ++i)
				state[i].store(0, std::memory_order_relaxed);
			for (int tid = 0; tid < n; ++tid) {
				workers.emplace_back([this, tid] {
					while (true) {
						int s;
						while ((s = state[tid].load(
								std::memory_order_acquire)) == 0)
							#if defined(__x86_64__) || defined(__i386__)
					__asm__ volatile("pause" ::: "memory");
#else
					std::this_thread::yield();
#endif
						if (s < 0)
							return;
						run_task(tasks[tid]);
						state[tid].store(0, std::memory_order_release);
					}
				});
			}
		}

		void dispatch() {
			for (int i = 0; i < nt; ++i)
				state[i].store(1, std::memory_order_release);
			for (int i = 0; i < nt; ++i)
				while (state[i].load(std::memory_order_acquire) != 0)
					#if defined(__x86_64__) || defined(__i386__)
					__asm__ volatile("pause" ::: "memory");
#else
					std::this_thread::yield();
#endif
		}

		~Pool() {
			for (int i = 0; i < nt; ++i)
				state[i].store(-1, std::memory_order_release);
			for (auto &w : workers)
				w.join();
		}
	};

	static int nt = std::min(64, std::max(1, (int)std::thread::hardware_concurrency()));
	static Pool pool(nt);

	int rows_per = N / nt;
	for (int tid = 0; tid < nt; ++tid) {
		pool.tasks[tid] = {
			a, b, c,
			tid * rows_per,
			(tid == nt - 1) ? N : (tid + 1) * rows_per,
			N
		};
	}
	pool.dispatch();

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

	// TILE=64: 3 tiles (A+B+C) × 16KB = 48KB → fits in typical 256KB L2 cache
	const int TILE = 64;
	int nt = std::min(64, std::max(1, (int)std::thread::hardware_concurrency()));
	int rows_per = N / nt;

	std::vector<std::thread> threads;
	threads.reserve(nt);

	for (int tid = 0; tid < nt; ++tid) {
		int i0 = tid * rows_per;
		int i1 = (tid == nt - 1) ? N : (tid + 1) * rows_per;

		threads.emplace_back([=] {
			// Zero the output rows this thread owns
			for (int i = i0; i < i1; ++i)
				for (int j = 0; j < N; ++j)
					c[(size_t)i * N + j] = 0.0f;

			// Tiled ikj GEMM
			// kk loop in 0,TILE,2*TILE,... order keeps k-accumulation
			// for each c[i][j] identical to sequential (0..N-1).
			for (int ii = i0; ii < i1; ii += TILE) {
				int ie = std::min(ii + TILE, i1);
				for (int kk = 0; kk < N; kk += TILE) {
					int ke = std::min(kk + TILE, N);
					for (int jj = 0; jj < N; jj += TILE) {
						int je = std::min(jj + TILE, N);
						for (int i = ii; i < ie; ++i) {
							for (int k = kk; k < ke; ++k) {
								float a_ik = a[(size_t)i * N + k];
								float *cij = c + (size_t)i * N + jj;
								const float *bkj = b + (size_t)k * N + jj;
								int width = je - jj;
								int j = 0;
								// 8-way unroll on j (independent stores)
								for (; j + 7 < width; j += 8) {
									cij[j]     += a_ik * bkj[j];
									cij[j + 1] += a_ik * bkj[j + 1];
									cij[j + 2] += a_ik * bkj[j + 2];
									cij[j + 3] += a_ik * bkj[j + 3];
									cij[j + 4] += a_ik * bkj[j + 4];
									cij[j + 5] += a_ik * bkj[j + 5];
									cij[j + 6] += a_ik * bkj[j + 6];
									cij[j + 7] += a_ik * bkj[j + 7];
								}
								for (; j < width; ++j)
									cij[j] += a_ik * bkj[j];
							}
						}
					}
				}
			}
		});
	}

	for (auto &t : threads)
		t.join();

	/****************/
}