#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
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

	const int T = std::max(1, std::min((int)std::thread::hardware_concurrency(), 16));
	std::vector<std::thread> threads;
	threads.reserve(T);
	int chunk = N / T;
	for (int t = 0; t < T; t++) {
		int start = t * chunk;
		int end   = (t == T - 1) ? N : start + chunk;
			threads.emplace_back([=]() {
				const double *b_end = b + N - 3;  // 내부 루프 종료 조건 사전 계산
				double *row0 = a + start * N;      // 외부 루프 곱셈 제거: 포인터 직접 증가
				double *row1 = row0 + N;
				int i = start;
				// 2행 동시 계산: b[j] 로드 1회로 두 행에 재사용
				for (; i <= end - 2; i += 2, row0 += 2*N, row1 += 2*N) {
					double s0=0, s1=0, s2=0, s3=0;
					double t0=0, t1=0, t2=0, t3=0;
					const double *pb  = b;     // 내부 루프 인덱스 연산 제거: 포인터 직접 증가
					const double *pr0 = row0;
					const double *pr1 = row1;
					for (; pb < b_end; pb+=4, pr0+=4, pr1+=4) {
						double bj0=pb[0], bj1=pb[1], bj2=pb[2], bj3=pb[3];
						s0 += pr0[0] * bj0;  t0 += pr1[0] * bj0;
						s1 += pr0[1] * bj1;  t1 += pr1[1] * bj1;
						s2 += pr0[2] * bj2;  t2 += pr1[2] * bj2;
						s3 += pr0[3] * bj3;  t3 += pr1[3] * bj3;
					}
					c[i]     = s0 + s1 + s2 + s3;
					c[i + 1] = t0 + t1 + t2 + t3;
				}
				// remainder: 홀수 행 처리
				for (; i < end; i++, row0 += N) {
					double s0=0, s1=0, s2=0, s3=0;
					const double *pb  = b;
					const double *pr0 = row0;
					for (; pb < b_end; pb+=4, pr0+=4) {
						s0 += pr0[0] * pb[0];
						s1 += pr0[1] * pb[1];
						s2 += pr0[2] * pb[2];
						s3 += pr0[3] * pb[3];
					}
					c[i] = s0 + s1 + s2 + s3;
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
