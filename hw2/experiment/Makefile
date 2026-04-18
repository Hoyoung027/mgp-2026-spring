build:
	mkdir -p build
	g++ -std=c++20 main.cpp -o build/main

run:
	./build/main

build_exp:
	mkdir -p build
	g++ -std=c++20 exp_all.cpp -o build/exp_all

run_exp:
	mkdir -p data
	./build/exp_all

run_exp1:
	mkdir -p data
	./build/exp_all 1

run_exp2:
	mkdir -p data
	./build/exp_all 2

run_exp3:
	mkdir -p data
	./build/exp_all 3

clean:
	rm -rf build

format:
	clang-format -i main.cpp parallel.h