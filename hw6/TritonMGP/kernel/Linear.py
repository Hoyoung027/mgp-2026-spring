import triton
import triton.language as tl
from mgp import empty


@triton.jit
def _linear_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
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
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


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
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
    )
    return out
