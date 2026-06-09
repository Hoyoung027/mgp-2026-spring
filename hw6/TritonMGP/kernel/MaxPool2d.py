import triton
import triton.language as tl
from mgp import empty


def triton_maxpool2d(
    x,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=(0, 0)
):
    """
    Applies a 2D max pooling operation over an input signal composed of several input planes.
    Args:
        x (torch.Tensor): Input tensor of shape (N, C, H, W) where N is the batch size, 
                          C is the number of channels, H is the height, and W is the width.
                          The input tensor must be on the GPU.
        kernel_size (tuple, optional): Size of the pooling window. Default is (2, 2).
        stride (tuple, optional): Stride of the pooling window. Default is (2, 2).
        padding (tuple, optional): Implicit zero padding to be added on both sides. Default is (0, 0).
        BLOCK_H (int, optional): Block height for Triton kernel. Default is 8.
        BLOCK_W (int, optional): Block width for Triton kernel. Default is 8.
    Returns:
        torch.Tensor: Output tensor after applying 2D max pooling, with shape (N, C, H_out, W_out).
    Raises:
        AssertionError: If the input tensor is not on the GPU.
    """
    n, c, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out = empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
    total = out.numel()

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    _maxpool2d_kernel[grid](
        x,
        out,
        total,
        c,
        h,
        w,
        out_h,
        out_w,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        BLOCK_SIZE=256,
    )
    return out


@triton.jit
def _maxpool2d_kernel(
    x_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % out_w
    oh = (offsets // out_w) % out_h
    ch = (offsets // (out_h * out_w)) % channels
    bn = offsets // (channels * out_h * out_w)

    acc = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    for rr in range(kernel_h):
        ih = oh * stride_h + rr - pad_h
        valid_h = (ih >= 0) & (ih < in_h)
        for ss in range(kernel_w):
            iw = ow * stride_w + ss - pad_w
            valid = mask & valid_h & (iw >= 0) & (iw < in_w)
            x_idx = ((bn * channels + ch) * in_h + ih) * in_w + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=-float("inf"))
            acc = tl.maximum(acc, val)

    tl.store(out_ptr + offsets, acc, mask=mask)
