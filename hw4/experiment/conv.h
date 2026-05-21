#include "cuda_runtime.h"

__global__ void im2col_float4_kernel(const float* __restrict__ x,
                                     int H, int W,
                                     int kH, int kW,
                                     int outH, int outW,
                                     float* __restrict__ out)
{
    int col_cols = outH * outW;
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec = (kH * kW * col_cols) >> 2;

    if (vec_idx >= total_vec) return;

    int elem = vec_idx << 2;
    int row_idx = elem / col_cols;
    int col_idx = elem - row_idx * col_cols;
    int kh = row_idx / kW;
    int kw = row_idx - kh * kW;

    float4 vals;

    int pos0 = col_idx;
    int oh0 = pos0 / outW;
    int ow0 = pos0 - oh0 * outW;
    vals.x = x[(oh0 + kh) * W + ow0 + kw];

    int pos1 = col_idx + 1;
    int oh1 = pos1 / outW;
    int ow1 = pos1 - oh1 * outW;
    vals.y = x[(oh1 + kh) * W + ow1 + kw];

    int pos2 = col_idx + 2;
    int oh2 = pos2 / outW;
    int ow2 = pos2 - oh2 * outW;
    vals.z = x[(oh2 + kh) * W + ow2 + kw];

    int pos3 = col_idx + 3;
    int oh3 = pos3 / outW;
    int ow3 = pos3 - oh3 * outW;
    vals.w = x[(oh3 + kh) * W + ow3 + kw];

    reinterpret_cast<float4*>(out)[vec_idx] = vals;
}

void launch_im2col(const float* x, int N, int C, int H, int W,
                   int kH, int kW, int padH, int padW, int strideH, int strideW,
                   int dilH, int dilW, float* out)
{
    int outH = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
    int outW = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;
    int total_vec = (C * kH * kW * outH * outW) >> 2;

    int block = 256;
    int grid = (total_vec + block - 1) / block;

    im2col_float4_kernel<<<grid, block>>>(x, H, W, kH, kW, outH, outW, out);
}

__global__ void matmul_one_row_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[k] * B[k * N + col];
    }

    C[col] = sum;
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K)
{
    int block = 256;
    int grid = (N + block - 1) / block;

    matmul_one_row_kernel<<<grid, block>>>(A, B, C, N, K);
}

__global__ void conv2d_direct_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ y,
                                     int H, int W,
                                     int kH, int kW,
                                     int outH, int outW)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (oh >= outH || ow >= outW) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            sum += x[(oh + kh) * W + ow + kw] * w[kh * kW + kw];
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

    conv2d_direct_kernel<<<grid, block>>>(x, w, y, H, W, kH, kW, outH, outW);
}
