#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
// You cannot use OpenMP <omp.h>
// Include header files if you need,
// but it must work without modifying the Makefile
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
		a[i] = (float)(rand()%2);
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
	for (int i = 0; i < N; ++i) {
		float sum = 0.0f;
		for (int k = 0; k < N; ++k)
			sum += a[i * N + k] * b[k];
		c[i] = sum;
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