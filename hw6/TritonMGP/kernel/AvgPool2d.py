import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _avgpool2d_kernel(
    x_ptr,
    out_ptr,
    num_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    q = offsets % Q
    p = (offsets // Q) % P
    c = (offsets // (P * Q)) % C
    n = offsets // (C * P * Q)

    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    for kh in range(0, KH):
        ih = p * SH + kh - PH
        h_mask = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = q * SW + kw - PW
            valid = mask & h_mask & (iw >= 0) & (iw < W)
            x_offsets = ((n * C + c) * H + ih) * W + iw
            acc += tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)

    acc = acc / (KH * KW)
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_avgpool2d(
    x,
    kernel_size=(7, 7),
    stride=(1, 1),
    padding=(0, 0),
    pool_size=None,
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
    if pool_size is not None:
        kernel_size = pool_size

    n, c, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    p = (h + 2 * ph - kh) // sh + 1
    q = (w + 2 * pw - kw) // sw + 1

    out = empty((n, c, p, q), device=x.device, dtype=x.dtype)
    n_elements = out.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _avgpool2d_kernel[grid](
        x,
        out,
        n_elements,
        c,
        h,
        w,
        p,
        q,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        BLOCK_SIZE=256,
    )
    return out
