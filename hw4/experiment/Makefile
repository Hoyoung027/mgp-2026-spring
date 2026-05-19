build:
	mkdir -p build
	/usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_86 -o build/main main.cu

run:
	./build/main

clean: 
	rm -rf build

format:
	clang-format -i main.cu
