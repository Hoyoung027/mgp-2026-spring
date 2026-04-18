#include "parallel-double-bench.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

static std::ofstream open_csv(const std::string &path, const std::string &header) {
    std::ifstream check(path);
    bool exists = check.good();
    check.close();
    std::ofstream f(path, std::ios::app);
    if (!exists) f << header << "\n";
    return f;
}

using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

static const int N    = 1 << 11;
static const int RUNS = 3;

static double *mat_a, *mat_b, *mat_c, *vec_b;

// ============================================================
// 실험 1: 스레드 수별 확장성 - 단일 (func, T) 측정
// usage: ./exp_all 1 gemv 8
//        ./exp_all 1 gemm 16
// ============================================================
void exp_thread_scaling(const std::string &func, int T) {
    std::cout << "\n=== [Experiment 1] Thread Scaling: " << func
              << " T=" << T << " (" << RUNS << " runs) ===\n";

    auto csv = open_csv("data/results_thread_scaling.csv", "func,threads,avg_sec");

    double total = 0.0;
    if (func == "gemv") {
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N];
            auto s = Clock::now();
            gemv_sse(mat_a, vec_b, out, N, T);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
    } else {
        for (int r = 0; r < RUNS; r++) {
            double *out = new double[N * N];
            auto s = Clock::now();
            gemm_sse(mat_a, mat_b, out, N, T);
            total += Sec(Clock::now() - s).count();
            delete[] out;
        }
    }
    double avg = total / RUNS;
    std::cout << std::fixed << std::setprecision(6) << "avg: " << avg << " sec\n";
    csv << func << "_sse," << T << "," << avg << "\n";
    std::cout << "  -> saved: data/results_thread_scaling.csv\n";
}

// ============================================================
// 실험 2: GEMM vs Freivalds 비교
// usage: ./exp_all 2
// ============================================================
void exp_gemm_vs_freivalds() {
    const int T_gemm = 32;
    const int T_gemv = 8;

    const long long ops_gemm      = 2LL * N * N * N;
    const long long ops_freivalds = 4LL * N * N;

    std::cout << "\n=== [Experiment 2] GEMM vs Freivalds (N=" << N
              << ", " << RUNS << " runs) ===\n";

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

    std::cout << std::fixed
              << "GEMM       ops: " << std::scientific << std::setprecision(2) << (double)ops_gemm
              << "  avg: " << std::fixed << std::setprecision(6) << avg_gemm << " sec\n"
              << "Freivalds  ops: " << std::scientific << std::setprecision(2) << (double)ops_freivalds
              << "  avg: " << std::fixed << std::setprecision(6) << avg_frv << " sec"
              << "  speedup: " << std::setprecision(1) << (avg_gemm / avg_frv) << "x\n";

    auto csv = open_csv("data/results_gemm_vs_freivalds.csv", "method,ops,avg_sec");
    csv << "gemm," << ops_gemm << "," << avg_gemm << "\n";
    csv << "freivalds," << ops_freivalds << "," << avg_frv << "\n";
    std::cout << "  -> saved: data/results_gemm_vs_freivalds.csv\n";
}

// ============================================================
// 실험 3: naive / unrolled / SSE 버전 비교 - 단일 func 측정
// usage: ./exp_all 3 gemv
//        ./exp_all 3 gemm
// ============================================================
void exp_versions(const std::string &func) {
    const int T_gemv = 8;
    const int T_gemm = 32;

    std::cout << "\n=== [Experiment 3] Version Comparison: " << func
              << " (" << RUNS << " runs) ===\n";

    auto csv = open_csv("data/results_versions.csv", "func,version,avg_sec");

    double t_naive, t_unrolled, t_sse;

    if (func == "gemv") {
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
    } else {
        {
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                double *out = new double[N * N];
                auto s = Clock::now();
                gemm_naive(mat_a, mat_b, out, N, T_gemm);
                total += Sec(Clock::now() - s).count();
                delete[] out;
            }
            t_naive = total / RUNS;
            std::cout << std::fixed << std::setprecision(6)
                      << "naive     avg: " << t_naive << " sec\n";
            csv << "gemm,naive," << t_naive << "\n";
        }
        {
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                double *out = new double[N * N];
                auto s = Clock::now();
                gemm_unrolled(mat_a, mat_b, out, N, T_gemm);
                total += Sec(Clock::now() - s).count();
                delete[] out;
            }
            t_unrolled = total / RUNS;
            std::cout << std::fixed << std::setprecision(6)
                      << "unrolled  avg: " << t_unrolled << " sec"
                      << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_unrolled) << "x\n";
            csv << "gemm,unrolled," << t_unrolled << "\n";
        }
        {
            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                double *out = new double[N * N];
                auto s = Clock::now();
                gemm_sse(mat_a, mat_b, out, N, T_gemm);
                total += Sec(Clock::now() - s).count();
                delete[] out;
            }
            t_sse = total / RUNS;
            std::cout << std::fixed << std::setprecision(6)
                      << "sse       avg: " << t_sse << " sec"
                      << "  speedup vs naive: " << std::setprecision(2) << (t_naive / t_sse) << "x\n";
            csv << "gemm,sse," << t_sse << "\n";
        }
    }
    std::cout << "  -> saved: data/results_versions.csv\n";
}

// ============================================================
// main
// usage: ./exp_all 1 gemv 8
//        ./exp_all 1 gemm 16
//        ./exp_all 2
//        ./exp_all 3 gemv
//        ./exp_all 3 gemm
// ============================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage:\n"
                  << "  ./exp_all 1 gemv <T>   # thread scaling, gemv\n"
                  << "  ./exp_all 1 gemm <T>   # thread scaling, gemm\n"
                  << "  ./exp_all 2            # GEMM vs Freivalds\n"
                  << "  ./exp_all 3 gemv       # version compare, gemv\n"
                  << "  ./exp_all 3 gemm       # version compare, gemm\n";
        return 1;
    }

    mat_a = new double[N * N];
    mat_b = new double[N * N];
    mat_c = new double[N * N];
    vec_b = new double[N];

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

    int exp = std::atoi(argv[1]);
    if (exp == 1) {
        std::string func = (argc > 2) ? argv[2] : "gemv";
        int T            = (argc > 3) ? std::atoi(argv[3]) : 8;
        exp_thread_scaling(func, T);
    } else if (exp == 2) {
        exp_gemm_vs_freivalds();
    } else if (exp == 3) {
        std::string func = (argc > 2) ? argv[2] : "gemv";
        exp_versions(func);
    }

    delete[] mat_a; delete[] mat_b; delete[] mat_c; delete[] vec_b;
    return 0;
}
