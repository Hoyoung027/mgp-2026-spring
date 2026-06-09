import triton
import triton.language as tl
from mgp import empty


def triton_avgpool2d(
    x,
    pool_size=(7, 7),
    stride=(1, 1),
    padding=(0, 0),
):
    """
    Applies a 2D average pooling over an input signal composed of several input planes.
    Args:
        x (torch.Tensor): Input tensor of shape [N, C, H, W].
        kernel_size (tuple, optional): Size of the pooling kernel. Default is (7, 7).
        stride (tuple, optional): Stride of the pooling operation. Default is (1, 1).
        padding (tuple, optional): Implicit zero padding to be added on both sides. Default is (0, 0).
    Returns:
        torch.Tensor: Output tensor after applying average pooling, with shape [N, C, 1, 1].
    """
    n, c, h, w = x.shape
    kh, kw = pool_size
    sh, sw = stride
    ph, pw = padding
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out = empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype)
    total = out.numel()

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    _avgpool2d_kernel[grid](
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
def _avgpool2d_kernel(
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

    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    count = tl.zeros((BLOCK_SIZE,), tl.float32)
    for rr in range(kernel_h):
        ih = oh * stride_h + rr - pad_h
        valid_h = (ih >= 0) & (ih < in_h)
        for ss in range(kernel_w):
            iw = ow * stride_w + ss - pad_w
            valid = mask & valid_h & (iw >= 0) & (iw < in_w)
            x_idx = ((bn * channels + ch) * in_h + ih) * in_w + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc += val
            count += tl.where(valid, 1.0, 0.0)

    tl.store(out_ptr + offsets, acc / count, mask=mask)
