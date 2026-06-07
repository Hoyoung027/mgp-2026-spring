import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _bn2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    num_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    channel = (offsets // (H * W)) % C

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel, mask=mask, other=1.0).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + eps) * gamma + beta
    tl.store(out_ptr + offsets, y, mask=mask)




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
    n_elements = input.numel()
    _, c, h, w = input.shape

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _bn2d_kernel[grid](
        input,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        n_elements,
        c,
        h,
        w,
        eps,
        BLOCK_SIZE=1024,
    )
    return out
