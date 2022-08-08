import torch
import torch.nn as nn
import numpy as np

class ResBlock_Module(nn.Module):
    """
    Residual Block for SR backbones

    """
    def __init__(self, num_conv, act_type, with_bn, ret_res_feat) -> None:
        super().__init__()
