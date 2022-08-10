# examples: RCAN

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualGroup(nn.Module):
    """
    -- block - block - block - ... - block -- + ->
    |_________________________________________|

        make several stacked blocks into a group
        refer to RCAN in BasicSR
        https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rcan_arch.py
    """
    def __init__(self, block, nb, nf, **kwargs):
        super(_ResidualGroup, self).__init__()
        self.res_group = nn.Sequential(
            *[block(nf=nf, **kwargs) for i in range(nb)]
        )
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1)
        
    def forward(self, x):
        res = self.conv(self.res_group(x))
        return res + x

class Groups_Struct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass