import triton
import triton.language as tl
from mgp import zeros


@triton.jit
def _conv2d_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_out_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    q = offs_m % Q
    p = (offs_m // Q) % P
    n = offs_m // (P * Q)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    offs_cks = tl.arange(0, BLOCK_K)
    for cks0 in range(0, C * R * S, BLOCK_K):
        cks = cks0 + offs_cks
        s = cks % S
        r = (cks // S) % R
        c = cks // (R * S)

        ih = p[:, None] * STRIDE_H + r[None, :] * DIL_H - PAD_H
        iw = q[:, None] * STRIDE_W + s[None, :] * DIL_W - PAD_W
        input_mask = (
            (offs_m[:, None] < M)
            & (cks[None, :] < C * R * S)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W)
        )
        input_offsets = ((n[:, None] * C + c[None, :]) * H + ih) * W + iw
        x = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        weight_mask = (offs_out_k[None, :] < K) & (cks[:, None] < C * R * S)
        weight_offsets = ((offs_out_k[None, :] * C + c[:, None]) * R + r[:, None]) * S + s[:, None]
        w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)

        acc += tl.dot(x, w)

    out_offsets = ((n[:, None] * K + offs_out_k[None, :]) * P + p[:, None]) * Q + q[:, None]
    out_mask = (offs_m[:, None] < M) & (offs_out_k[None, :] < K)
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_conv2d(input, weight, bias, stride, padding, dilation):
    """
    Perform a 2D convolution operation using Triton.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W) where
            N is the batch size, C is the number of input channels,
            H is the height, and W is the width.
        weight (torch.Tensor): Weight tensor of shape (K, C, R, S) where
            K is the number of output channels, C is the number of input channels,
            R is the height of the convolution kernel, and S is the width of the convolution kernel.
        bias (torch.Tensor): Bias tensor of shape (K,).
        stride (tuple): Stride of the convolution (str_h, str_w).
        padding (tuple): Padding added to both sides of the input (pad_h, pad_w).
        dilation (tuple): Spacing between kernel elements (dil_h, dil_w).

    Returns:
        torch.Tensor: Output tensor of shape (N, K, P, Q) where
            N is the batch size, K is the number of output channels,
            P is the height of the output, and Q is the width of the output.
    """
    n, c, h, w = input.shape
    k, _, r, s = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    p = (h + 2 * pad_h - dil_h * (r - 1) - 1) // stride_h + 1
    q = (w + 2 * pad_w - dil_w * (s - 1) - 1) // stride_w + 1

    out = zeros((n, k, p, q), device=input.device, dtype=input.dtype)
    m = n * p * q

    grid = (triton.cdiv(m, 32), triton.cdiv(k, 32))
    _conv2d_kernel[grid](
        input,
        weight,
        out,
        m,
        k,
        c,
        h,
        w,
        p,
        q,
        r,
        s,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return out
