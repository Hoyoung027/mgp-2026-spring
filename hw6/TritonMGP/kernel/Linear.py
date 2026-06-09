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

    m, k = a.shape
    _, n = b.shape
    out = empty((m, n), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    _linear_kernel[grid](
        a,
        b,
        bias,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=64,
    )
    return out


@triton.jit
def _linear_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, k, BLOCK_K):
        kk = k0 + offs_k
        a_vals = tl.load(
            a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
            mask=(offs_m[:, None] < m) & (kk[None, :] < k),
            other=0.0,
        )
        b_vals = tl.load(
            b_ptr + kk[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(kk[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc += tl.dot(a_vals, b_vals)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < n, other=0.0)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )
