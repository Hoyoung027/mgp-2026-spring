#include "parallel-double.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdlib.h>

using namespace std;

bool is_same_vec(double *a, double *b, int N) {
	for (int i = 0; i < N; i++) {
		if (abs(a[i] - b[i]) > 1e-2) {
			cout << std::setprecision(17) << "Mismatch at (" << i << "): " << a[i]
				 << " vs " << b[i] << " (diff=" << (a[i] - b[i]) << ")" << endl;
			return false;
		}
	}
	return true;
}

bool is_same_mat(double *a, double *b, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (abs(a[i * N + j] - b[i * N + j]) > 1e-3) {
				cout << std::setprecision(17) << "Mismatch at (" << i << ", " << j
					 << "): " << a[i * N + j] << " vs " << b[i * N + j]
					 << " (diff=" << (a[i * N + j] - b[i * N + j]) << ")" << endl;
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char **argv) {
	const int N = 1 << 11;

	double *mat_a = new double[N * N];
	double *mat_b = new double[N * N];
	double *vec_b = new double[N];
	double *mat_c = new double[N * N];
	double *vec_c = new double[N];

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
		std::chrono::duration<double> diff;
		auto start = std::chrono::steady_clock::now();
		{
			double *gemm_out = new double[N * N];
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
		double *my_vec = new double[N];
		init_vec(my_vec, N);

		std::chrono::duration<double> diff;
		auto start = std::chrono::steady_clock::now();
		{
			double *gemv_out1 = new double[N];
			double *gemv_out2 = new double[N];
			double *buf = new double[N];

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
