import triton
import triton.language as tl
from mgp import empty, empty_like




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
