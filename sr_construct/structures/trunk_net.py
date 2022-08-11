# examples: ECBSR/IMDN/ELAN

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.basic import PixelShuffle_Module

class Trunk_Struct(nn.Module):
    """
             |----------------------------------------|
    x - conv - [block-block-...-block] - fusion_conv -+- pixelshuffle -
                  |____|_____|____|__________|
    """
    def __init__(self, in_nc, out_nc, struct_nf, struct_nb, scale, block, **kwargs) -> None:
        super(Trunk_Struct, self).__init__()
        self.nb = struct_nb
        self.conv_in = nn.Conv2d(in_nc, struct_nf, 3, 1, 1)
        self.trunk = nn.ModuleList(
            [block(**kwargs) for _ in range(struct_nb)]
        )
        self.fusion_conv = nn.Conv2d(struct_nf * struct_nb, struct_nf, 1, 1, 0)
        self.upsampler = PixelShuffle_Module(struct_nb, out_nc, scale)
    
    def forward(self, x):
        feat_in = self.conv_in(x)
        outB_ls = []
        outB = self.trunk[0](feat_in)
        outB_ls.append(outB)
        for i in range(1, self.nb):
            outB = self.trunk[i](outB)
            outB_ls.append(outB)
        out_cat = torch.cat(outB_ls)
        fused = self.fusion_conv(out_cat)
        sr_out = self.upsampler(fused)
        return sr_out