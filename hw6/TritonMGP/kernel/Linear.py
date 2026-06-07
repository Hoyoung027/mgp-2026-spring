import triton
import triton.language as tl
from mgp import empty


def triton_linear(a, b, bias):
    """
    Perform a linear transformation using Triton.
    Args:
        a (torch.Tensor): The first input matrix (M x K) which must be contiguous.
        b (torch.Tensor): The second input matrix (K x N) which must be contiguous.
        bias (torch.Tensor): The bias vector (N) which must be compatible with the second dimension of matrix b.
    Returns:
        torch.Tensor: The result of the linear transformation (M x N).
    Raises:
        AssertionError: If any of the input matrices are not contiguous or if their dimensions are incompatible.
    """
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert b.shape[1] == bias.shape[0], "Incompatible bias dimensions"

    