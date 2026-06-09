import triton
import triton.language as tl
from mgp import empty




def triton_bn2d(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    momentum: float = 0.1,
    eps: float = 1e-5
    ) :
    """
    Applies a 2D batch normalization using Triton.
    Args:
        input (Tensor): Input tensor of shape (N, C, H, W).
        weight (Tensor): Scale tensor of shape (C,).
        bias (Tensor): Bias tensor of shape (C,).
        running_mean (Tensor): Running mean tensor of shape (C,).
        running_var (Tensor): Running variance tensor of shape (C,).
        momentum (float, optional): Value used for the running_mean and running_var computation. Default: 0.1.
        eps (float, optional): A value added to the denominator for numerical stability. Default: 1e-5.
    Returns:
        Tensor: The normalized and optionally activated output tensor of the same shape as input.
    """
    out = empty(input.shape, device=input.device, dtype=input.dtype)
    n, c, h, w = input.shape
    total = input.numel()

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    _bn2d_kernel[grid](
        input,
        out,
        weight,
        bias,
        running_mean,
        running_var,
        total,
        c,
        h * w,
        eps,
        BLOCK_SIZE=1024,
    )
    return out


@triton.jit
def _bn2d_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    c = (offsets // spatial) % channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)

    y = (x - mean) * tl.rsqrt(var + eps) * weight + bias
    tl.store(out_ptr + offsets, y, mask=mask)
