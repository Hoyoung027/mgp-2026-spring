import triton
import triton.language as tl
from mgp import empty



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
    out_h = (h + 2 * pad_h - dil_h * (r - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dil_w * (s - 1) - 1) // stride_w + 1
    out = empty((n, k, out_h, out_w), device=input.device, dtype=input.dtype)

    if r == 1 and s == 1 and pad_h == 0 and pad_w == 0:
        grid = (triton.cdiv(n * out_h * out_w, 64), triton.cdiv(k, 32))
        _conv1x1_kernel[grid](
            input,
            weight,
            out,
            n * out_h * out_w,
            c,
            k,
            h,
            w,
            out_h,
            out_w,
            stride_h,
            stride_w,
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
    else:
        grid = (triton.cdiv(n * out_h * out_w, 32), triton.cdiv(k, 32))
        _conv2d_kernel[grid](
            input,
            weight,
            out,
            n * out_h * out_w,
            c,
            k,
            h,
            w,
            out_h,
            out_w,
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
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
    return out


@triton.jit
def _conv1x1_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    total_m: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ow = offs_m % out_w
    oh = (offs_m // out_w) % out_h
    bn = offs_m // (out_h * out_w)
    ih = oh * stride_h
    iw = ow * stride_w

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for c0 in range(0, in_channels, BLOCK_K):
        cc = c0 + offs_k
        x_idx = ((bn[:, None] * in_channels + cc[None, :]) * in_h + ih[:, None]) * in_w + iw[:, None]
        w_idx = (offs_n[None, :] * in_channels + cc[:, None])

        x_vals = tl.load(
            x_ptr + x_idx,
            mask=(offs_m[:, None] < total_m) & (cc[None, :] < in_channels),
            other=0.0,
        )
        w_vals = tl.load(
            w_ptr + w_idx,
            mask=(offs_n[None, :] < out_channels) & (cc[:, None] < in_channels),
            other=0.0,
        )
        acc += tl.dot(x_vals, w_vals)

    out_idx = (bn[:, None] * out_channels + offs_n[None, :]) * out_h * out_w + oh[:, None] * out_w + ow[:, None]
    tl.store(
        out_ptr + out_idx,
        acc,
        mask=(offs_m[:, None] < total_m) & (offs_n[None, :] < out_channels),
    )


@triton.jit
def _conv2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    total_m: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dil_h: tl.constexpr,
    dil_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    reduction = in_channels * kernel_h * kernel_w

    ow = offs_m % out_w
    oh = (offs_m // out_w) % out_h
    bn = offs_m // (out_h * out_w)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, reduction, BLOCK_K):
        kk = k0 + offs_k
        rs = kk % (kernel_h * kernel_w)
        rr = rs // kernel_w
        ss = rs % kernel_w
        cc = kk // (kernel_h * kernel_w)

        ih = oh[:, None] * stride_h + rr[None, :] * dil_h - pad_h
        iw = ow[:, None] * stride_w + ss[None, :] * dil_w - pad_w
        x_idx = ((bn[:, None] * in_channels + cc[None, :]) * in_h + ih) * in_w + iw
        x_mask = (
            (offs_m[:, None] < total_m)
            & (kk[None, :] < reduction)
            & (ih >= 0)
            & (ih < in_h)
            & (iw >= 0)
            & (iw < in_w)
        )
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        w_idx = ((offs_n[None, :] * in_channels + cc[:, None]) * kernel_h + rr[:, None]) * kernel_w + ss[:, None]
        w_vals = tl.load(
            w_ptr + w_idx,
            mask=(offs_n[None, :] < out_channels) & (kk[:, None] < reduction),
            other=0.0,
        )
        acc += tl.dot(x_vals, w_vals)

    out_idx = (bn[:, None] * out_channels + offs_n[None, :]) * out_h * out_w + oh[:, None] * out_w + ow[:, None]
    tl.store(
        out_ptr + out_idx,
        acc,
        mask=(offs_m[:, None] < total_m) & (offs_n[None, :] < out_channels),
    )
