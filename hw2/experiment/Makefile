build:
	mkdir -p build
	g++ -std=c++20 main.cpp -o build/main

run:
	./build/main

build_exp:
	mkdir -p build
	g++ -std=c++20 exp_all.cpp -o build/exp_all

run_exp2:
	mkdir -p data
	./build/exp_all 2

# 실험 1: make run_exp1 FUNC=gemv T=8
run_exp1:
	mkdir -p data
	./build/exp_all 1 $(FUNC) $(T)

# 실험 3: make run_exp3 FUNC=gemv  or  make run_exp3 FUNC=gemm
run_exp3:
	mkdir -p data
	./build/exp_all 3 $(FUNC)

clean:
	rm -rf build

format:
	clang-format -i main.cpp parallel.h