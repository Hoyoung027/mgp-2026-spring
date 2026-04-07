#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
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

			for (int i = i_start; i < i_end; ++i) {
				// prefetch next row into cache (cache line = 64 bytes = 16 floats)
				if (i + 1 < i_end) {
					const float *ai_next = a + (i + 1) * N;
					for (int p = 0; p < N; p += 16)
						__builtin_prefetch(ai_next + p, 0, 0);
				}

				float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
				float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
				const float *ai = a + i * N;
				int k = 0;
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
				float sum = (sum0 + sum1) + (sum2 + sum3) +
				            (sum4 + sum5) + (sum6 + sum7);
				for (; k < N; ++k)
					sum += ai[k] * b[k];
				c[i] = sum;
			}
		});
	}

	for (auto &t : threads) t.join();

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
