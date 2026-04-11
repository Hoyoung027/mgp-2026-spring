#include "parallel.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdlib.h>

using namespace std;

bool is_same_vec(float *a, float *b, int N) {
	for (int i = 0; i < N; i++) {
		if (abs(a[i] - b[i]) > 1e-7) {
			return false;
		}
	}
	return true;
}

bool is_same_mat(float *a, float *b, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (abs(a[i * N + j] - b[i * N + j]) > 1e-7) {
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char **argv) {
	const int N = 1 << 11;

	float *mat_a = new float[N * N];
	float *mat_b = new float[N * N];
	float *vec_b = new float[N];
	float *mat_c = new float[N * N];
	float *vec_c = new float[N];

	fstream f_mat_a("/data/hw2/matrix_a.txt", ios::in);
	fstream f_mat_b("/data/hw2/matrix_b.txt", ios::in);
	fstream f_vec_b("/data/hw2/vector_b.txt", ios::in);
	fstream f_mat_c("/data/hw2/matrix_c.txt", ios::in);
	fstream f_vec_c("/data/hw2/vector_c.txt", ios::in);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			f_mat_a >> mat_a[i * N + j];
			f_mat_b >> mat_b[i * N + j];
			f_mat_c >> mat_c[i * N + j];
		}
		f_vec_b >> vec_b[i];
		f_vec_c >> vec_c[i];
	}

	// 1. Parallel GEMM Approach
	{
		std::chrono::duration<float> diff;
		auto start = std::chrono::steady_clock::now();
		{
			float *gemm_out = new float[N * N];
			gemm(mat_a, mat_b, gemm_out, N);
			if (!is_same_mat(gemm_out, mat_c, N)) {
				cerr << "Parallel GEMM Approach Failed" << endl;
				return -1;
			}
		}
		auto end = std::chrono::steady_clock::now();
		diff = end - start;
		std::cout << "Parallel GEMM Approach took " << diff.count() << " sec"
				  << std::endl;
		cout << "Parallel GEMM Approach Passed" << endl;
	}

	// 2. Parallel Freivalds’ Algorithm
	{
		float *my_vec = new float[N];
		init_vec(my_vec, N);

		std::chrono::duration<float> diff;
		auto start = std::chrono::steady_clock::now();
		{
			float *gemv_out1 = new float[N];
			float *gemv_out2 = new float[N];
			float *buf = new float[N];

			gemv(mat_b, my_vec, buf, N);
			gemv(mat_a, buf, gemv_out1, N);
			gemv(mat_c, my_vec, gemv_out2, N);

			if (!is_same_vec(gemv_out1, gemv_out2, N)) {
				cerr << "Freivalds' Algorithm Failed" << endl;
				return -1;
			}
		}
		auto end = std::chrono::steady_clock::now();
		diff = end - start;
		std::cout << "Freivalds' Algorithm took " << diff.count() << " sec"
				  << std::endl;
		cout << "Freivalds' Algorithm Passed" << endl;
	}

	return 0;
}
