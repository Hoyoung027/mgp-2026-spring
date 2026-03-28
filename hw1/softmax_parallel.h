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

	/****************/
}