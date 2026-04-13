#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <thread> // added
#include <vector> // added
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile

/**
 * @brief Initializes a vector with random double values.
 *
 * This function fills the given array
 *
 * @param a Pointer to the array to be initialized.
 * @param N The number of elements in the array.
 */
inline void init_vec(double *a, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

	srand(42);
	for (int i = 0; i < N; ++i)
        a[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

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
inline void gemv(double *a, double *b, double *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

	struct GemvPool {
		int T = 0;
		double *a = nullptr, *b = nullptr, *c = nullptr;
		int N = 0;
		std::mutex mtx;
		std::condition_variable cv_work, cv_done;
		int task_id    = 0;
		int done_count = 0;
		bool shutdown  = false;
		std::vector<std::thread> threads;
	};
	static GemvPool pool;
	static std::once_flag init_flag;

	std::call_once(init_flag, []() {
		pool.T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 32));
		for (int t = 1; t < pool.T; t++) {
			pool.threads.emplace_back([t]() {
				int my_last = 0;
				while (true) {
					double *wa, *wb, *wc; int wN, wT;
					{
						std::unique_lock<std::mutex> lk(pool.mtx);
						pool.cv_work.wait(lk, [&my_last]{
							return pool.task_id > my_last || pool.shutdown;
						});
						if (pool.shutdown) break;
						my_last = pool.task_id;
						wa = pool.a; wb = pool.b; wc = pool.c;
						wN = pool.N; wT = pool.T;
					}
					int chunk = wN / wT;
					int start = t * chunk;
					int end   = (t == wT - 1) ? wN : start + chunk;
					for (int i = start; i < end; i++) {
						double *row = wa + i * wN;
						double s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
						int j = 0;
						for (; j <= wN - 8; j += 8) {
							s0 += row[j]     * wb[j];
							s1 += row[j + 1] * wb[j + 1];
							s2 += row[j + 2] * wb[j + 2];
							s3 += row[j + 3] * wb[j + 3];
							s4 += row[j + 4] * wb[j + 4];
							s5 += row[j + 5] * wb[j + 5];
							s6 += row[j + 6] * wb[j + 6];
							s7 += row[j + 7] * wb[j + 7];
						}
						for (; j < wN; j++) s0 += row[j] * wb[j];
						wc[i] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
					}
					{
						std::lock_guard<std::mutex> lk(pool.mtx);
						if (++pool.done_count == pool.T - 1)
							pool.cv_done.notify_one();
					}
				}
			});
		}
	});

	// 작업 설정 후 워커 깨우기
	{
		std::lock_guard<std::mutex> lk(pool.mtx);
		pool.a = a; pool.b = b; pool.c = c; pool.N = N;
		pool.done_count = 0;
		pool.task_id++;
		pool.cv_work.notify_all();
	}

	// main thread가 tid=0 구간 직접 계산
	{
		int chunk = N / pool.T;
		for (int i = 0; i < chunk; i++) {
			double *row = a + i * N;
			double s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
			int j = 0;
			for (; j <= N - 8; j += 8) {
				s0 += row[j]     * b[j];
				s1 += row[j + 1] * b[j + 1];
				s2 += row[j + 2] * b[j + 2];
				s3 += row[j + 3] * b[j + 3];
				s4 += row[j + 4] * b[j + 4];
				s5 += row[j + 5] * b[j + 5];
				s6 += row[j + 6] * b[j + 6];
				s7 += row[j + 7] * b[j + 7];
			}
			for (; j < N; j++) s0 += row[j] * b[j];
			c[i] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
		}
	}

	// 워커 완료 대기
	{
		std::unique_lock<std::mutex> lk(pool.mtx);
		pool.cv_done.wait(lk, []{ return pool.done_count == pool.T - 1; });
	}

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
inline void gemm(double *a, double *b, double *c, int N) {
	/****************/
	/* TODO: put your own parallelized code here */
	/* You don't have to parallelize all of your code - it's up to you. */

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 32));
	std::vector<std::thread> threads;
	threads.reserve(T);
	int chunk = N / T;
	for (int t = 0; t < T; t++) {
		int start = t * chunk;
		int end   = (t == T - 1) ? N : start + chunk;
		threads.emplace_back([=]() {
			for (int i = start; i < end; i++)
				for (int j = 0; j < N; j++) c[i * N + j] = 0.0;
			for (int i = start; i < end; i++)
				for (int k = 0; k < N; k++) {
					double aik = a[i * N + k];
					int j = 0;
					for (; j <= N - 4; j += 4) {
						c[i * N + j]     += aik * b[k * N + j];
						c[i * N + j + 1] += aik * b[k * N + j + 1];
						c[i * N + j + 2] += aik * b[k * N + j + 2];
						c[i * N + j + 3] += aik * b[k * N + j + 3];
					}
					for (; j < N; j++) c[i * N + j] += aik * b[k * N + j];
				}
		});
	}
	for (auto &th : threads) th.join();

	/****************/
}
