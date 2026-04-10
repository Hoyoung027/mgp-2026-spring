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
using clk = std::chrono::steady_clock;
using sec = std::chrono::duration<double>;

bool is_same_vec(float *a, float *b, int N) {
    for (int i = 0; i < N; i++)
        if (a[i] != b[i]) return false;
    return true;
}

bool is_same_mat(float *a, float *b, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (a[i*N+j] != b[i*N+j]) return false;
    return true;
}

// -----------------------------------------------
// 실험: 스레드 생성 비용 단독 측정
// -----------------------------------------------
double measure_thread_creation(int n_threads) {
    auto t0 = clk::now();
    {
        std::vector<std::thread> tmp;
        tmp.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i)
            tmp.emplace_back([] { std::this_thread::yield(); });
        for (auto &t : tmp) t.join();
    }
    auto t1 = clk::now();
    return sec(t1 - t0).count();
}

int main(int argc, char **argv) {
    const int N = 1 << 11;  // 2048

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
            f_mat_a >> mat_a[i*N+j];
            f_mat_b >> mat_b[i*N+j];
            f_mat_c >> mat_c[i*N+j];
        }
        f_vec_b >> vec_b[i];
        f_vec_c >> vec_c[i];
    }

    cout << "========================================" << endl;
    cout << "  Thread Creation Cost Experiment" << endl;
    cout << "  N = " << N << ", target threads = 16" << endl;
    cout << "========================================" << endl;

    // -----------------------------------------------
    // [실험 A] 스레드 생성 비용만 측정 (compute 없음)
    // -----------------------------------------------
    cout << "\n[A] Thread Creation Cost Only (no compute)" << endl;
    for (int n : {1, 4, 8, 16, 32, 64}) {
        double t = measure_thread_creation(n);
        cout << "  " << n << " threads created+joined: "
             << t * 1000.0 << " ms"
             << "  (avg " << t * 1000000.0 / n << " us/thread)" << endl;
    }

    // -----------------------------------------------
    // [실험 B] Per-call thread (매번 생성/삭제) × 3회
    // -----------------------------------------------
    cout << "\n[B] Per-call thread creation (Freivalds x3, 16 threads each)" << endl;
    {
        float *my_vec = new float[N];
        init_vec(my_vec, N);
        float *out1 = new float[N], *out2 = new float[N], *buf = new float[N];
        g_pool = nullptr;  // per-call mode

        auto t0 = clk::now();
        gemv(mat_b, my_vec, buf,  N);
        gemv(mat_a, buf,    out1, N);
        gemv(mat_c, my_vec, out2, N);
        auto t1 = clk::now();

        bool ok = is_same_vec(out1, out2, N);
        cout << "  Total: " << sec(t1-t0).count() * 1000.0 << " ms"
             << "  [" << (ok ? "PASS" : "FAIL") << "]" << endl;
        delete[] my_vec; delete[] out1; delete[] out2; delete[] buf;
    }

    // -----------------------------------------------
    // [실험 C] Pool inside timer (static과 동일한 효과 시뮬레이션)
    //          → 타이머 안에서 pool 생성 후 3회 gemv
    // -----------------------------------------------
    cout << "\n[C] Pool created INSIDE timer (simulates static GemvPool)" << endl;
    {
        float *my_vec = new float[N];
        init_vec(my_vec, N);
        float *out1 = new float[N], *out2 = new float[N], *buf = new float[N];

        auto t0 = clk::now();
        {
            GemvPool pool(16);   // ← 타이머 안에서 생성
            g_pool = &pool;
            gemv(mat_b, my_vec, buf,  N);
            gemv(mat_a, buf,    out1, N);
            gemv(mat_c, my_vec, out2, N);
            g_pool = nullptr;
        }  // pool 소멸 (스레드 join)
        auto t1 = clk::now();

        bool ok = is_same_vec(out1, out2, N);
        cout << "  Total: " << sec(t1-t0).count() * 1000.0 << " ms"
             << "  [" << (ok ? "PASS" : "FAIL") << "]" << endl;
        delete[] my_vec; delete[] out1; delete[] out2; delete[] buf;
    }

    // -----------------------------------------------
    // [실험 D] Pool created OUTSIDE timer
    //          → 타이머 전에 pool 생성, 3회 gemv만 측정
    // -----------------------------------------------
    cout << "\n[D] Pool created OUTSIDE timer (pure compute only)" << endl;
    {
        float *my_vec = new float[N];
        init_vec(my_vec, N);
        float *out1 = new float[N], *out2 = new float[N], *buf = new float[N];

        // 타이머 밖에서 pool 생성
        auto pool_t0 = clk::now();
        GemvPool pool(16);
        auto pool_t1 = clk::now();
        cout << "  Pool creation time: "
             << sec(pool_t1-pool_t0).count() * 1000.0 << " ms" << endl;

        g_pool = &pool;

        auto t0 = clk::now();
        gemv(mat_b, my_vec, buf,  N);
        gemv(mat_a, buf,    out1, N);
        gemv(mat_c, my_vec, out2, N);
        auto t1 = clk::now();

        g_pool = nullptr;

        bool ok = is_same_vec(out1, out2, N);
        cout << "  Compute only: " << sec(t1-t0).count() * 1000.0 << " ms"
             << "  [" << (ok ? "PASS" : "FAIL") << "]" << endl;
        cout << "  Total (pool+compute): "
             << sec(pool_t1-pool_t0).count() * 1000.0 + sec(t1-t0).count() * 1000.0
             << " ms" << endl;

        delete[] my_vec; delete[] out1; delete[] out2; delete[] buf;
    }

    // -----------------------------------------------
    // [실험 E] 각 gemv 호출별 breakdown (pool outside)
    // -----------------------------------------------
    cout << "\n[E] Per-gemv breakdown with pre-created pool" << endl;
    {
        float *my_vec = new float[N];
        init_vec(my_vec, N);
        float *out1 = new float[N], *out2 = new float[N], *buf = new float[N];

        GemvPool pool(16);
        g_pool = &pool;

        auto t0 = clk::now();
        gemv(mat_b, my_vec, buf,  N);
        auto t1 = clk::now();
        gemv(mat_a, buf,    out1, N);
        auto t2 = clk::now();
        gemv(mat_c, my_vec, out2, N);
        auto t3 = clk::now();

        g_pool = nullptr;

        cout << "  gemv #1 (B*v):   " << sec(t1-t0).count() * 1000.0 << " ms" << endl;
        cout << "  gemv #2 (A*buf): " << sec(t2-t1).count() * 1000.0 << " ms" << endl;
        cout << "  gemv #3 (C*v):   " << sec(t3-t2).count() * 1000.0 << " ms" << endl;
        cout << "  Total:           " << sec(t3-t0).count() * 1000.0 << " ms" << endl;

        delete[] my_vec; delete[] out1; delete[] out2; delete[] buf;
    }

    cout << "\n========================================" << endl;
    cout << "  Summary" << endl;
    cout << "========================================" << endl;
    cout << "  B: Per-call threads = thread_creation × 3 + compute × 3" << endl;
    cout << "  C: Pool inside timer = pool_creation + compute × 3" << endl;
    cout << "  D: Pool outside timer = compute × 3 only" << endl;
    cout << "  => C - D = pool creation cost" << endl;
    cout << "  => B - D = per-call thread creation cost × 3" << endl;

    return 0;
}
