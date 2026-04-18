#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <thread> // added
#include <vector> // added
#include <immintrin.h> // added
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

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 8));
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
					__m128d aik_v = _mm_set1_pd(a[i * N + k]);
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

	/****************/
}
