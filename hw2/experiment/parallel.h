#include <algorithm>
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

	int num_threads = (int)std::thread::hardware_concurrency();
	if (num_threads < 1) num_threads = 1;
	if (num_threads > 64) num_threads = 64;

	// b를 int16으로 quantize (한 번만, 모든 스레드 공유)
	float max_b = 1e-9f;
	for (int k = 0; k < N; k++)
		if (std::abs(b[k]) > max_b) max_b = std::abs(b[k]);
	const float scale_b     = 32767.0f / max_b;
	const float inv_scale_b = max_b    / 32767.0f;

	std::vector<int16_t> bq(N);
	for (int k = 0; k < N; k++)
		bq[k] = (int16_t)(b[k] * scale_b + (b[k] >= 0 ? 0.5f : -0.5f));

	std::vector<std::thread> threads;
	threads.reserve(num_threads);

	for (int tid = 0; tid < num_threads; ++tid) {
		threads.emplace_back([=, &bq]() {
			int rows_per = N / num_threads;
			int i_start  = tid * rows_per;
			int i_end    = (tid == num_threads - 1) ? N : i_start + rows_per;

			std::vector<int16_t> aq(N); // 행 quantization 버퍼 (스레드당 1개)

			for (int i = i_start; i < i_end; ++i) {
				const float *ai = a + i * N;

				// 행의 max 탐색 → scale 계산
				float max_row = 1e-9f;
				for (int k = 0; k < N; k++)
					if (std::abs(ai[k]) > max_row) max_row = std::abs(ai[k]);
				const float scale_row     = 32767.0f / max_row;
				const float inv_scale_row = max_row    / 32767.0f;

				// 행을 int16으로 quantize
				for (int k = 0; k < N; k++)
					aq[k] = (int16_t)(ai[k] * scale_row + (ai[k] >= 0 ? 0.5f : -0.5f));

				// int16 dot product: _mm_madd_epi16
				// 8개 int16 쌍을 곱한 뒤 인접 쌍을 더해 4개 int32로 누적
				__m128i vsum = _mm_setzero_si128();
				int k = 0;
				for (; k <= N - 8; k += 8) {
					__m128i va = _mm_loadu_si128((const __m128i*)(aq.data() + k));
					__m128i vb = _mm_loadu_si128((const __m128i*)(bq.data() + k));
					vsum = _mm_add_epi32(vsum, _mm_madd_epi16(va, vb));
				}

				// horizontal reduction: 4개 int32 → 1개
				__m128i tmp = _mm_shuffle_epi32(vsum, _MM_SHUFFLE(1,0,3,2));
				vsum = _mm_add_epi32(vsum, tmp);
				tmp  = _mm_shuffle_epi32(vsum, _MM_SHUFFLE(2,3,0,1));
				vsum = _mm_add_epi32(vsum, tmp);
				int32_t dot = _mm_cvtsi128_si32(vsum);

				// 나머지 처리
				for (; k < N; k++)
					dot += (int32_t)aq[k] * (int32_t)bq[k];

				// dequantize
				c[i] = (float)dot * inv_scale_row * inv_scale_b;
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

	// ── Step 1: Transpose B → BT (병렬 + 타일링) ──────────────────────────
	// BT[j][k] = B[k][j]: C[i][j] = dot(A[i], BT[j]) 로 둘 다 sequential 접근
	float *bt = new float[(size_t)N * N];
	{
		std::vector<std::thread> tp;
		tp.reserve(num_threads);
		for (int tid = 0; tid < num_threads; ++tid) {
			tp.emplace_back([=]() {
				int rows_per = N / num_threads;
				int j_start  = tid * rows_per;
				int j_end    = (tid == num_threads - 1) ? N : j_start + rows_per;
				// 타일링으로 B의 column 접근 cache miss 완화
				for (int jj = j_start; jj < j_end; jj += TILE) {
					int j_lim = std::min(jj + TILE, j_end);
					for (int kk = 0; kk < N; kk += TILE) {
						int k_lim = std::min(kk + TILE, N);
						for (int j = jj; j < j_lim; ++j)
							for (int k = kk; k < k_lim; ++k)
								bt[j * N + k] = b[k * N + j];
					}
				}
			});
		}
		for (auto &t : tp) t.join();
	}

	// ── Step 2: Tiled GEMM with BT ──────────────────────────────────────────
	// 루프 순서: ii → jj → kk → i → j → k
	// 각 (i,j)에 대해 k가 0,1,...,N-1 순서로 누적 → float 결과 동일 보장
	{
		std::vector<std::thread> threads;
		threads.reserve(num_threads);
		for (int tid = 0; tid < num_threads; ++tid) {
			threads.emplace_back([=]() {
				int rows_per = N / num_threads;
				int i_start  = tid * rows_per;
				int i_end    = (tid == num_threads - 1) ? N : i_start + rows_per;

				for (int i = i_start; i < i_end; ++i)
					for (int j = 0; j < N; ++j)
						c[i * N + j] = 0.0f;

				for (int ii = i_start; ii < i_end; ii += TILE) {
					int i_lim = std::min(ii + TILE, i_end);
					for (int jj = 0; jj < N; jj += TILE) {
						int j_lim = std::min(jj + TILE, N);
						for (int kk = 0; kk < N; kk += TILE) {
							int k_lim = std::min(kk + TILE, N);
							for (int i = ii; i < i_lim; ++i) {
								const float *ai  = a  + i * N;
								for (int j = jj; j < j_lim; ++j) {
									const float *btj = bt + j * N;
									// A[i][k]와 BT[j][k] 모두 sequential 접근
									for (int k = kk; k < k_lim; ++k)
										c[i * N + j] += ai[k] * btj[k];
								}
							}
						}
					}
				}
			});
		}
		for (auto &t : threads) t.join();
	}

	delete[] bt;

	/****************/
}
