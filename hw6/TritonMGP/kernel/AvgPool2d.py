import triton
import triton.language as tl
from mgp import empty


def triton_avgpool2d(
    x,
    kernel_size=(7, 7),
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
   