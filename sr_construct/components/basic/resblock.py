import torch
import torch.nn as nn
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_act_layer


class ResBlock_Module(nn.Module):
    """
    Residual Block for SR backbones
    x ---[conv-(bn)-act]-[conv-(bn)-act]- ... -[conv-(bn)]-+- (act) -->
    |______________________________________________________|
    """
    def __init__(self, num_conv, n_feat, ksize, 
                act_type='relu', with_bn=False, 
                with_last_act=True, ret_res_feat=False):
        super(ResBlock_Module, self).__init__()
        block_ls = []
        assert ksize % 2 == 1
        padding = ksize // 2
        for layer_id in range(num_conv):
            block_ls.append(nn.Conv2d(n_feat, n_feat, ksize, 1, padding))
            if with_bn:
                block_ls.append(nn.BachNorm2d(n_feat))
            if layer_id != num_conv - 1:
                block_ls.append(get_act_layer(act_type, n_feat))
        # body part of resblock
        self.body = nn.Sequential(*block_ls)
        self.with_last_act = with_last_act
        if with_last_act:
            self.last_act = get_act_layer(act_type, n_feat)
        self.ret_res_feat = ret_res_feat

    def forward(self, x):
        res = self.body(x)
        if self.with_last_act:
            out = self.last_act(res + x)
        else:
            out = res + x
        if self.ret_res_feat:
            return out, res
        else:
            return out