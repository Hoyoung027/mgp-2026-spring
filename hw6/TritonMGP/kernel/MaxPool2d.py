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
