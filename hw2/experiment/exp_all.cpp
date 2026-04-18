#include "parallel-double-bench.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static const int N    = 1 << 11;
static const int RUNS = 5;

static double *mat_a, *mat_b, *mat_c, *vec_b, *tmp1, *tmp2, *tmp3;

// ============================================================
// 실험 1: 스레드 수별 확장성 (gemv_sse, gemm_sse)
// ============================================================
void exp_thread_scaling() {
    const int thread_counts_gemv[] = {1, 2, 4, 8, 16, 32, 64};
    const int thread_counts_gemm[] = {2, 4, 8, 16, 32, 64};
    const int n_tc_gemv = sizeof(thread_counts_gemv) / sizeof(thread_counts_gemv[0]);
    const int n_tc_gemm = sizeof(thread_counts_gemm) / sizeof(thread_counts_gemm[0]);
    std::cout << "\n=== [Experiment 1] Thread Scaling ===\n";

    std::ofstream csv("data/results_thread_scaling.csv");
    csv << "func,threads,avg_sec\n";

    std::cout << "\n-- gemv_sse --\n";
    for (int ti = 0; ti < n_tc_gemv; ti++) {
        int T = thread_counts_gemv[ti];
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N];
            auto s = Clock::now();
            gemv_sse(mat_a, vec_b, out, N, T);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
        double avg = total / RUNS;
        std::cout << std::fixed << std::setprecision(6)
                  << "T=" << std::setw(2) << T << "  avg: " << avg << " sec\n";
        csv << "gemv_sse," << T << "," << avg << "\n";
    }

    std::cout << "\n-- gemm_sse (T>=4 only) --\n";
    for (int ti = 0; ti < n_tc_gemm; ti++) {
        int T = thread_counts_gemm[ti];
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N * N];
            auto s = Clock::now();
            gemm_sse(mat_a, mat_b, out, N, T);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
        double avg = total / RUNS;
        std::cout << std::fixed << std::setprecision(3)
                  << "T=" << std::setw(2) << T << "  avg: " << avg << " sec\n";
        csv << "gemm_sse," << T << "," << avg << "\n";
    }

    std::cout << "  -> saved: data/results_thread_scaling.csv\n";
}

// ============================================================
// 실험 2: GEMM vs Freivalds 비교
// ============================================================
void exp_gemm_vs_freivalds() {
    const int T_gemm = 32;
    const int T_gemv = 8;

    const long long ops_gemm      = 2LL * N * N * N;
    const long long ops_freivalds = 4LL * N * N;

    std::cout << "\n=== [Experiment 2] GEMM vs Freivalds (N=" << N
              << ", " << RUNS << " runs) ===\n";

    // init_vec은 타이머 밖에서 수행 (main.cpp와 동일)
    double *my_vec = new double[N];
    srand(42);
    for (int i = 0; i < N; i++)
        my_vec[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);

    double total_gemm = 0.0;
    for (int r = 0; r < RUNS; r++) {
        double *out = new double[N * N];
        auto s = Clock::now();
        gemm_sse(mat_a, mat_b, out, N, T_gemm);
        total_gemm += Sec(Clock::now() - s).count();
        delete[] out;
    }
    double avg_gemm = total_gemm / RUNS;

    double total_frv = 0.0;
    for (int r = 0; r < RUNS; r++) {
        double *out1 = new double[N];
        double *out2 = new double[N];
        double *buf  = new double[N];
        auto s = Clock::now();
        gemv_sse(mat_b, my_vec, buf,  N, T_gemv);
        gemv_sse(mat_a, buf,    out1, N, T_gemv);
        gemv_sse(mat_c, my_vec, out2, N, T_gemv);
        total_frv += Sec(Clock::now() - s).count();
        delete[] out1; delete[] out2; delete[] buf;
    }
    double avg_frv = total_frv / RUNS;
    delete[] my_vec;

    std::cout << std::fixed << std::setprecision(3)
              << "GEMM       ops: " << std::scientific << std::setprecision(2) << (double)ops_gemm
              << "  avg: " << std::fixed << std::setprecision(3) << avg_gemm << " sec\n"
              << "Freivalds  ops: " << std::scientific << std::setprecision(2) << (double)ops_freivalds
              << "  avg: " << std::fixed << std::setprecision(3) << avg_frv * 1e3 << " ms"
              << "  speedup: " << std::setprecision(1) << (avg_gemm / avg_frv) << "x\n";

    std::ofstream csv("data/results_gemm_vs_freivalds.csv");
    csv << "method,ops,avg_sec\n";
    csv << "gemm," << ops_gemm << "," << avg_gemm << "\n";
    csv << "freivalds," << ops_freivalds << "," << avg_frv << "\n";
    std::cout << "  -> saved: data/results_gemm_vs_freivalds.csv\n";
}

// ============================================================
// 실험 3: naive / unrolled / SSE 버전 비교
// ============================================================
void exp_versions() {
    const int T_gemv = 8;
    const int T_gemm = 32;

    std::cout << "\n=== [Experiment 3] Version Comparison ===\n";

    std::ofstream csv("data/results_versions.csv");
    csv << "func,version,avg_sec\n";

    double t_naive, t_unrolled, t_sse;

    // gemv
    std::cout << "\n-- gemv (T=" << T_gemv << ", " << RUNS << " runs) --\n";
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N];
            auto s = Clock::now();
            gemv_naive(mat_a, vec_b, out, N, T_gemv);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
        t_naive = total / RUNS;
        std::cout << std::fixed << std::setprecision(6)
                  << "naive     avg: " << t_naive << " sec\n";
        csv << "gemv,naive," << t_naive << "\n";
    }
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N];
            auto s = Clock::now();
            gemv_unrolled(mat_a, vec_b, out, N, T_gemv);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
        t_unrolled = total / RUNS;
        std::cout << std::fixed << std::setprecision(6)
                  << "unrolled  avg: " << t_unrolled << " sec"
                  << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_unrolled) << "x\n";
        csv << "gemv,unrolled," << t_unrolled << "\n";
    }
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N];
            auto s = Clock::now();
            gemv_sse(mat_a, vec_b, out, N, T_gemv);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
        t_sse = total / RUNS;
        std::cout << std::fixed << std::setprecision(6)
                  << "sse       avg: " << t_sse << " sec"
                  << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_sse) << "x\n";
        csv << "gemv,sse," << t_sse << "\n";
    }

    // gemm
    std::cout << "\n-- gemm (T=" << T_gemm << ", " << RUNS << " runs) --\n";
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            auto s = Clock::now();
            gemm_naive(mat_a, mat_b, tmp2, N, T_gemm);
            total += Sec(Clock::now() - s).count();
        }
        t_naive = total / RUNS;
        std::cout << std::fixed << std::setprecision(3)
                  << "naive     avg: " << t_naive << " sec\n";
        csv << "gemm,naive," << t_naive << "\n";
    }
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            auto s = Clock::now();
            gemm_unrolled(mat_a, mat_b, tmp2, N, T_gemm);
            total += Sec(Clock::now() - s).count();
        }
        t_unrolled = total / RUNS;
        std::cout << std::fixed << std::setprecision(3)
                  << "unrolled  avg: " << t_unrolled << " sec"
                  << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_unrolled) << "x\n";
        csv << "gemm,unrolled," << t_unrolled << "\n";
    }
    {
        double total = 0.0;
        for (int r = 0; r < RUNS; r++) {
            auto s = Clock::now();
            gemm_sse(mat_a, mat_b, tmp2, N, T_gemm);
            total += Sec(Clock::now() - s).count();
        }
        t_sse = total / RUNS;
        std::cout << std::fixed << std::setprecision(3)
                  << "sse       avg: " << t_sse << " sec"
                  << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_sse) << "x\n";
        csv << "gemm,sse," << t_sse << "\n";
    }

    std::cout << "  -> saved: data/results_versions.csv\n";
}

// ============================================================
// main
// ============================================================
int main(int argc, char **argv) {
    mat_a = new double[N * N];
    mat_b = new double[N * N];
    mat_c = new double[N * N];
    vec_b = new double[N];
    tmp1  = new double[N];
    tmp2  = new double[N * N];
    tmp3  = new double[N];

    std::fstream f_a("/data/hw2/matrix_a.txt", std::ios::in);
    std::fstream f_b("/data/hw2/matrix_b.txt", std::ios::in);
    std::fstream f_c("/data/hw2/matrix_c.txt", std::ios::in);
    std::fstream f_v("/data/hw2/vector_b.txt", std::ios::in);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f_a >> mat_a[i*N+j];
            f_b >> mat_b[i*N+j];
            f_c >> mat_c[i*N+j];
        }
        f_v >> vec_b[i];
    }

    int exp = (argc > 1) ? std::atoi(argv[1]) : 0;
    switch (exp) {
        case 1: exp_thread_scaling();     break;
        case 2: exp_gemm_vs_freivalds();  break;
        case 3: exp_versions();           break;
        default:
            exp_thread_scaling();
            exp_gemm_vs_freivalds();
            exp_versions();
    }

    delete[] mat_a; delete[] mat_b; delete[] mat_c;
    delete[] vec_b; delete[] tmp1;  delete[] tmp2; delete[] tmp3;
    return 0;
}
