#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "reduction.h"

#define REDUCE_BLOCK_SIZE 256
#define REDUCE_MAX_BLOCKS 64

static inline int div_up_int(int x, int y)
{
    return (x + y - 1) / y;
}

__global__ void reduce0(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce1(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        unsigned int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce2(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce3(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    int sum = 0;
    if (i < n) {
        sum += g_idata[i];
    }
    if (i + blockDim.x < n) {
        sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce4(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    int sum = 0;
    if (i < n) {
        sum += g_idata[i];
    }
    if (i + blockDim.x < n) {
        sum += g_idata[i + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce5(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    int sum = 0;
    if (i < n) {
        sum += g_idata[i];
    }
    if (i + blockSize < n) {
        sum += g_idata[i + blockSize];
    }
    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int* smem = sdata;
        if (blockSize >= 64) { smem[tid] += smem[tid + 32]; }
        if (blockSize >= 32) { smem[tid] += smem[tid + 16]; }
        if (blockSize >= 16) { smem[tid] += smem[tid + 8]; }
        if (blockSize >= 8) { smem[tid] += smem[tid + 4]; }
        if (blockSize >= 4) { smem[tid] += smem[tid + 2]; }
        if (blockSize >= 2) { smem[tid] += smem[tid + 1]; }
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce6(const int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    int sum = 0;

    while (i < n) {
        sum += g_idata[i];
        if (i + blockSize < n) {
            sum += g_idata[i + blockSize];
        }
        i += gridSize;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int* smem = sdata;
        if (blockSize >= 64) { smem[tid] += smem[tid + 32]; }
        if (blockSize >= 32) { smem[tid] += smem[tid + 16]; }
        if (blockSize >= 16) { smem[tid] += smem[tid + 8]; }
        if (blockSize >= 8) { smem[tid] += smem[tid + 4]; }
        if (blockSize >= 4) { smem[tid] += smem[tid + 2]; }
        if (blockSize >= 2) { smem[tid] += smem[tid + 1]; }
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int launch_reduce0(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads);
    reduce0<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce1(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads);
    reduce1<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce2(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads);
    reduce2<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce3(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads * 2);
    reduce3<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce4(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads * 2);
    reduce4<<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce5(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads * 2);
    reduce5<REDUCE_BLOCK_SIZE><<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

int launch_reduce6(const int* d_idata, int* d_odata, int n)
{
    int threads = REDUCE_BLOCK_SIZE;
    int blocks = div_up_int(n, threads * 2);
    if (blocks > REDUCE_MAX_BLOCKS) {
        blocks = REDUCE_MAX_BLOCKS;
    }
    reduce6<REDUCE_BLOCK_SIZE><<<blocks, threads, threads * sizeof(int)>>>(d_idata, d_odata, n);
    return blocks;
}

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}


void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}


void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    (void)g_idata;
    (void)g_odata;

    int partial_count = launch_reduce6(d_idata, d_odata, n);

    while (partial_count > 1) {
        partial_count = launch_reduce6(d_odata, d_odata, partial_count);
    }
}
