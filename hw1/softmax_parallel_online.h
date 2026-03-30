#include <algorithm>
#include <cmath>
#include <limits>
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
	float local_sum[NTHREADS]; // exp 보정 누적합 (online softmax의 d값)
	std::thread threads[NTHREADS];

	// Phase 1 (병렬): Online Softmax
	// 각 스레드가 담당 구간에서 max와 sum을 한 번의 pass로 동시에 계산
	// 일반적인 방법은 max 탐색 후 exp 계산이지만, online softmax는
	// max가 갱신될 때마다 기존 합을 exp(m_old - m_new)로 보정하면서 진행
	for (int t = 0; t < NTHREADS; t++) {
		int start = (long long)t * elems / NTHREADS;
		int end   = (long long)(t + 1) * elems / NTHREADS;
		threads[t] = std::thread([=, &local_max, &local_sum]() {
			float m = -std::numeric_limits<float>::infinity();
			float d = 0.0f;
			for (int i = start; i < end; i++) {
				float x = in[i];
				float m_new = std::max(m, x);
				// 이전 합(d)을 새로운 max 기준으로 보정 후 현재 원소 추가
				d = d * std::exp(m - m_new) + std::exp(x - m_new);
				m = m_new;
			}
			local_max[t] = m;
			local_sum[t] = d;
		});
	}
	for (int t = 0; t < NTHREADS; t++) threads[t].join();

	// Reduction: 각 스레드의 (local_max, local_sum)을 하나로 합침
	// 두 구간 (m1,d1), (m2,d2)를 합칠 때도 동일한 보정 공식 적용
	float global_max = local_max[0];
	float global_sum = local_sum[0];
	for (int t = 1; t < NTHREADS; t++) {
		float m_new = std::max(global_max, local_max[t]);
		global_sum = global_sum * std::exp(global_max - m_new)
		           + local_sum[t] * std::exp(local_max[t] - m_new);
		global_max = m_new;
	}
	float inv_sum = 1.0f / global_sum; // 나눗셈 대신 역수 곱셈으로 성능 개선

	// Phase 2 (병렬): 최종 softmax 값 계산
	// global_max와 inv_sum이 확정된 후 각 원소에 대해 exp(x - max) * inv_sum
	for (int t = 0; t < NTHREADS; t++) {
		int start = (long long)t * elems / NTHREADS;
		int end   = (long long)(t + 1) * elems / NTHREADS;
		threads[t] = std::thread([=]() {
			for (int i = start; i < end; i++) {
				out[i] = std::exp(in[i] - global_max) * inv_sum;
			}
		});
	}
	for (int t = 0; t < NTHREADS; t++) threads[t].join();

	/****************/
}