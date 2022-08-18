# examples: RCAN

from lib2to3.pgen2.pgen import NFAState
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.basic.rescale import ProgPixelShuffle_Module, PixelShuffle_Module

class _ResidualGroup(nn.Module):
    """
    -- block - block - ... - block -- conv -- + ->
    |_________________________________________|

        make several stacked blocks into a group
        refer to RCAN in BasicSR
        https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rcan_arch.py
    """
    def __init__(self, block, rg_nb, rg_nf, **kwargs):
        super(_ResidualGroup, self).__init__()
        self.res_group = nn.Sequential(
            *[block(nf=rg_nf, **kwargs) for i in range(rg_nb)]
        )
        self.conv = nn.Conv2d(rg_nf, rg_nf, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.res_group(x))
        return res + x

class Groups_Struct(nn.Module):
    def __init__(self, in_nc, out_nc, block, rg_nb, rg_nf, scale, num_groups, **kwargs):
        super(Groups_Struct, self).__init__()
        self.conv_in = nn.Conv2d(in_nc, rg_nf, 3, 1, 1)
        group_ls = []
        for _ in range(num_groups):
            group_ls.append(_ResidualGroup(block, rg_nb, rg_nf, **kwargs))
        self.body = nn.Sequential(group_ls)
        if scale & (scale - 1) == 0:
            self.upsampler = ProgPixelShuffle_Module(rg_nf, out_nc, scale)
        else:
            self.upsampler = PixelShuffle_Module(rg_nf, out_nc, scale)

    def forward(self, x):
        feat_in = self.conv_in(x)
        feat_out = self.body(feat_in)
        out = self.upsampler(feat_out)
        return out