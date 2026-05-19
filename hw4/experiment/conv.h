#include "cuda_runtime.h"

// ============================================================
// Naive im2col
// ============================================================

__global__ void im2col_kernel(
    const float* x, int C, int H, int W,
    int kH, int kW, int padH, int padW,
    int strideH, int strideW, int dilH, int dilW,
    int outH, int outW, float* out)
{
    int col_cols = outH * outW;
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int total    = C * kH * kW * col_cols;

    if (tid >= total) return;

    // col_idx varies fastest → coalesced writes & reads
    int col_idx = tid % col_cols;
    int row_idx = tid / col_cols;  // = c * kH * kW + kh * kW + kw

    int oh  = col_idx / outW;
    int ow  = col_idx % outW;
    int kw  = row_idx % kW;
    int kh  = (row_idx / kW) % kH;
    int c   = row_idx / (kH * kW);

    int ih = oh * strideH - padH + kh * dilH;
    int iw = ow * strideW - padW + kw  * dilW;

    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        val = x[(c * H + ih) * W + iw];

    out[row_idx * col_cols + col_idx] = val;
}

void launch_im2col(const float* x, int N, int C, int H, int W,
                   int kH, int kW, int padH, int padW, int strideH, int strideW,
                   int dilH, int dilW, float* out)
{
    int outH  = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int outW  = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;
    int total = C * kH * kW * outH * outW;

    int block = 256;
    int grid  = (total + block - 1) / block;

    im2col_kernel<<<grid, block>>>(
        x, C, H, W, kH, kW, padH, padW,
        strideH, strideW, dilH, dilW, outH, outW, out);
}

// ============================================================
// Naive Matmul: C = A * B,  A(M×K), B(K×N), C(M×N)
// ============================================================

__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

__global__ void matmul_one_row_kernel(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[k] * B[k * N + col];

    C[col] = sum;
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    if (M == 1) {
        int block = 256;
        int grid = (N + block - 1) / block;

        matmul_one_row_kernel<<<grid, block>>>(A, B, C, N, K);
        return;
    }

    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// ============================================================
// Naive Direct Conv2d
// ============================================================

__global__ void conv2d_direct_kernel(
    const float* x, const float* w, float* y,
    int N, int C, int H, int W,
    int K, int kH, int kW,
    int padH, int padW, int strideH, int strideW,
    int dilH, int dilW, int outH, int outW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (oh >= outH || ow >= outW) return;

    // N=1, K=1, C=1 in graded input, but written generically
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float val = 0.0f;
            for (int c = 0; c < C; c++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * strideH - padH + kh * dilH;
                        int iw = ow * strideW - padW + kw * dilW;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            val += x[((n * C + c) * H + ih) * W + iw]
                                 * w[((k * C + c) * kH + kh) * kW + kw];
                    }
                }
            }
            y[((n * K + k) * outH + oh) * outW + ow] = val;
        }
    }
}

template<int R>
__global__ void conv2d_direct_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int H, int W, int outH, int outW)
{
    extern __shared__ float tile[];

    int tileW = blockDim.x + R - 1;
    int tileH = blockDim.y + R - 1;
    int baseX = blockIdx.x * blockDim.x;
    int baseY = blockIdx.y * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads = blockDim.x * blockDim.y;

    for (int i = tid; i < tileW * tileH; i += threads) {
        int ly = i / tileW;
        int lx = i - ly * tileW;
        int ix = baseX + lx;
        int iy = baseY + ly;

        tile[i] = (iy < H && ix < W) ? x[iy * W + ix] : 0.0f;
    }
    __syncthreads();

    int ow = baseX + threadIdx.x;
    int oh = baseY + threadIdx.y;

    if (oh >= outH || ow >= outW) return;

    float sum = 0.0f;
#pragma unroll
    for (int kh = 0; kh < R; kh++) {
#pragma unroll
        for (int kw = 0; kw < R; kw++) {
            sum += tile[(threadIdx.y + kh) * tileW + threadIdx.x + kw]
                 * w[kh * R + kw];
        }
    }

    y[oh * outW + ow] = sum;
}

void launch_conv2d_direct(const float* x, const float* w, float* y,
                          int N, int C, int H, int W,
                          int K, int kH, int kW,
                          int padH, int padW, int strideH, int strideW,
                          int dilH, int dilW)
{
    int outH = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int outW = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

    dim3 block(32, 8);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y);

    if (N == 1 && C == 1 && K == 1 &&
        padH == 0 && padW == 0 &&
        strideH == 1 && strideW == 1 &&
        dilH == 1 && dilW == 1 && kH == kW) {
        if (kH == 3) {
            size_t shmem = (block.x + 2) * (block.y + 2) * sizeof(float);
            conv2d_direct_tiled_kernel<3><<<grid, block, shmem>>>(
                x, w, y, H, W, outH, outW);
            return;
        }
        if (kH == 7) {
            size_t shmem = (block.x + 6) * (block.y + 6) * sizeof(float);
            conv2d_direct_tiled_kernel<7><<<grid, block, shmem>>>(
                x, w, y, H, W, outH, outW);
            return;
        }
    }

    conv2d_direct_kernel<<<grid, block>>>(
        x, w, y, N, C, H, W, K, kH, kW,
        padH, padW, strideH, strideW, dilH, dilW, outH, outW);
}
