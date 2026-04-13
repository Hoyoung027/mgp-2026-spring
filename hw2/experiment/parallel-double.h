#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <emmintrin.h>  // SSE2
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

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 32));
	std::vector<std::thread> threads;
	threads.reserve(T);
	int chunk = N / T;
	for (int t = 0; t < T; t++) {
		int start = t * chunk;
		int end   = (t == T - 1) ? N : start + chunk;
			threads.emplace_back([=]() {
				for (int i = start; i < end; i++) {
					double *row = a + i * N;
					__m128d acc0 = _mm_setzero_pd();
					__m128d acc1 = _mm_setzero_pd();
					__m128d acc2 = _mm_setzero_pd();
					__m128d acc3 = _mm_setzero_pd();
					int j = 0;
					for (; j <= N - 8; j += 8) {
						acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(&row[j]),     _mm_loadu_pd(&b[j])));
						acc1 = _mm_add_pd(acc1, _mm_mul_pd(_mm_loadu_pd(&row[j + 2]), _mm_loadu_pd(&b[j + 2])));
						acc2 = _mm_add_pd(acc2, _mm_mul_pd(_mm_loadu_pd(&row[j + 4]), _mm_loadu_pd(&b[j + 4])));
						acc3 = _mm_add_pd(acc3, _mm_mul_pd(_mm_loadu_pd(&row[j + 6]), _mm_loadu_pd(&b[j + 6])));
					}
					acc0 = _mm_add_pd(acc0, acc1);
					acc2 = _mm_add_pd(acc2, acc3);
					acc0 = _mm_add_pd(acc0, acc2);
					acc0 = _mm_add_pd(acc0, _mm_shuffle_pd(acc0, acc0, 1));
					double sum = _mm_cvtsd_f64(acc0);
					for (; j < N; j++) sum += row[j] * b[j];
					c[i] = sum;
				}
			});
	}
	for (auto &th : threads) th.join();

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
