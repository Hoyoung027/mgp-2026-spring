#include <cmath>
#include <cstdint>
#include <iostream>

// ARM system counter (고정 1 GHz 타이머)
static uint64_t read_counter() {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

int main() {
    const int N = 10000000;
    volatile float sink = 0;

    // warmup
    for (int i = 0; i < 10000; i++) sink = std::exp(1.5f);

    // exp() 측정
    uint64_t start = read_counter();
    for (int i = 0; i < N; i++) sink = std::exp(1.5f + i * 0.000001f);
    uint64_t end = read_counter();
    double exp_ticks = (double)(end - start) / N;

    // add() 측정
    start = read_counter();
    for (int i = 0; i < N; i++) sink = 1.5f + (i * 0.000001f);
    end = read_counter();
    double add_ticks = (double)(end - start) / N;

    // mul() 측정
    start = read_counter();
    for (int i = 0; i < N; i++) sink = 1.5f * (i * 0.000001f + 1.0f);
    end = read_counter();
    double mul_ticks = (double)(end - start) / N;

    // div() 측정
    start = read_counter();
    for (int i = 0; i < N; i++) sink = 1.5f / (i * 0.000001f + 1.0f);
    end = read_counter();
    double div_ticks = (double)(end - start) / N;

    std::cout << "=== exp() cost benchmark ===" << std::endl;
    std::cout << "N = " << N << " iterations" << std::endl;
    std::cout << std::endl;
    std::cout << "exp() : " << exp_ticks << " ticks/op" << std::endl;
    std::cout << "add() : " << add_ticks << " ticks/op" << std::endl;
    std::cout << "mul() : " << mul_ticks << " ticks/op" << std::endl;
    std::cout << "div() : " << div_ticks << " ticks/op" << std::endl;
    std::cout << std::endl;
    std::cout << "exp / add ratio: " << exp_ticks / add_ticks << "x" << std::endl;
    std::cout << "exp / mul ratio: " << exp_ticks / mul_ticks << "x" << std::endl;
    std::cout << "exp / div ratio: " << exp_ticks / div_ticks << "x" << std::endl;

    return 0;
}