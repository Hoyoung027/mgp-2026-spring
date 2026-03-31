#include <algorithm>
#include <cmath>
#include <thread>
// Do NOT add any other headers.

using namespace std; // You can remove this line if you want.

/**
 * @brief Computes the softmax function in parallel.
 *
 * This function takes an input array of floats and computes the softmax
 * function in parallel using the specified number of threads. The result
 * is stored in the output array.
 *
 * @param in Pointer to the input array of floats.
 * @param out Pointer to the output array where the softmax results will be
 * stored.
 * @param elems The number of elements in the input array.
 */
inline void softmax_parallel(float *in, float *out, int elems) {

	const int NTHREADS = 16; // Number of threads to use.
							 // Yon can change it to any number you want.

	/****************/
	/* TODO: put your own parallelized softmax here */
	/* You don't have to parallelize all of your code - it's up to you. */

	float local_max[NTHREADS];
	float local_sum[NTHREADS];
	std::thread threads[NTHREADS];

	// Phase 1 (병렬): 구간별 max 탐색
	for (int t = 0; t < NTHREADS; t++) {
		int start = (long long)t * elems / NTHREADS;
		int end   = (long long)(t + 1) * elems / NTHREADS;
		threads[t] = std::thread([=, &local_max]() {
			float m = in[start];
			for (int i = start + 1; i < end; i++)
				m = std::max(m, in[i]);
			local_max[t] = m;
		});
	}
	for (int t = 0; t < NTHREADS; t++) threads[t].join();

	// Reduction: global_max 확정
	float global_max = local_max[0];
	for (int t = 1; t < NTHREADS; t++)
		global_max = std::max(global_max, local_max[t]);

	// Phase 2 (병렬): exp 계산 + 구간별 합산
	for (int t = 0; t < NTHREADS; t++) {
		int start = (long long)t * elems / NTHREADS;
		int end   = (long long)(t + 1) * elems / NTHREADS;
		threads[t] = std::thread([=, &local_sum]() {
			float s = 0.0f;
			for (int i = start; i < end; i++) {
				out[i] = std::exp(in[i] - global_max);
				s += out[i];
			}
			local_sum[t] = s;
		});
	}
	for (int t = 0; t < NTHREADS; t++) threads[t].join();

	// Reduction: global_sum 확정
	float global_sum = 0.0f;
	for (int t = 0; t < NTHREADS; t++)
		global_sum += local_sum[t];
	float inv_sum = 1.0f / global_sum;

	// Phase 3 (병렬): 나누기
	for (int t = 0; t < NTHREADS; t++) {
		int start = (long long)t * elems / NTHREADS;
		int end   = (long long)(t + 1) * elems / NTHREADS;
		threads[t] = std::thread([=]() {
			for (int i = start; i < end; i++)
				out[i] *= inv_sum;
		});
	}
	for (int t = 0; t < NTHREADS; t++) threads[t].join();

	/****************/
}