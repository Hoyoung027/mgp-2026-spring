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
		a[i] = (double)(rand() & 1);

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

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 16));
	std::vector<std::thread> threads;
	threads.reserve(T);
	int chunk = N / T;
	for (int t = 0; t < T; t++) {
		int start = t * chunk;
		int end   = (t == T - 1) ? N : start + chunk;
		threads.emplace_back([=]() {
			for (int i = start; i < end; i++) {
				double sum = 0.0;
				for (int j = 0; j < N; j++) sum += a[i * N + j] * b[j];
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

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 16));
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
					for (int j = 0; j < N; j++) c[i * N + j] += aik * b[k * N + j];
				}
		});
	}
	for (auto &th : threads) th.join();

	/****************/
}
