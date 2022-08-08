import torch
import torch.nn as nn
import numpy as np

class ResBlock_Module(nn.Module):
    """
    A unified residual block module for SR archs

        arch_code (int): 'CAC' # C:conv, A:act, B:bn
        act_type (str): default 'relu', one of 'relu', 'lrelu', 'elu'
        res_weight (float): default 1.0
        ret_res_feat (bool): default True, refer to RFANet

    """
    def __init__(self, num_conv, act_type, with_bn, ret_res_feat) -> None:
        super().__init__()
