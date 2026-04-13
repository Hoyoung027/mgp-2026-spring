#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <pthread.h>
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

	struct Args { double *a, *b, *c; int N, start, end; };
	struct Worker {
		static void *run(void *arg) {
			auto *p = static_cast<Args *>(arg);
			for (int i = p->start; i < p->end; ++i) {
				double sum = 0.0;
				for (int j = 0; j < p->N; ++j)
					sum += p->a[i * p->N + j] * p->b[j];
				p->c[i] = sum;
			}
			return nullptr;
		}
	};

	int nt = std::min((int)std::thread::hardware_concurrency(), 32);
	std::vector<Args> args(nt);
	std::vector<pthread_t> threads(nt);
	int chunk = (N + nt - 1) / nt;

	for (int t = 0; t < nt; ++t) {
		args[t] = {a, b, c, N, t * chunk, std::min((t + 1) * chunk, N)};
		pthread_create(&threads[t], nullptr, Worker::run, &args[t]);
	}
	for (int t = 0; t < nt; ++t)
		pthread_join(threads[t], nullptr);

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

	struct Args { double *a, *b, *c; int N, start, end; };
	struct Worker {
		static void *run(void *arg) {
			constexpr int TILE = 48;
			auto *p = static_cast<Args *>(arg);
			double *a = p->a, *b = p->b, *c = p->c;
			int N = p->N;

			for (int i = p->start; i < p->end; ++i)
				for (int j = 0; j < N; ++j)
					c[i * N + j] = 0.0;

			for (int ii = p->start; ii < p->end; ii += TILE)
				for (int kk = 0; kk < N; kk += TILE)
					for (int jj = 0; jj < N; jj += TILE)
						for (int i = ii; i < std::min(ii + TILE, p->end); ++i)
							for (int k = kk; k < std::min(kk + TILE, N); ++k) {
								double aik = a[i * N + k];
								int jend = std::min(jj + TILE, N);
								for (int j = jj; j < jend; ++j)
									c[i * N + j] += aik * b[k * N + j];
							}
			return nullptr;
		}
	};

	int nt = (int)std::thread::hardware_concurrency();
	std::vector<Args> args(nt);
	std::vector<pthread_t> threads(nt);
	int chunk = (N + nt - 1) / nt;

	for (int t = 0; t < nt; ++t) {
		args[t] = {a, b, c, N, t * chunk, std::min((t + 1) * chunk, N)};
		pthread_create(&threads[t], nullptr, Worker::run, &args[t]);
	}
	for (int t = 0; t < nt; ++t)
		pthread_join(threads[t], nullptr);

	/****************/
}
