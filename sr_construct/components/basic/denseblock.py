import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# modified from densenet-pytorch
# https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py

class _DenseBasicBlock(nn.Module):
    """
    - (bn) - act - conv
    """
    def __init__(self, in_feat, out_feat, with_bn=False, act_type='relu'):
        super(_DenseBasicBlock, self).__init__()
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(in_feat)
        self.act = get_act_layer(act_type, in_feat)
        self.conv = nn.Conv2d(in_feat, out_feat, 3, 3, 1, bias=False)
    
    def forward(self, x):
        if self.with_bn:
            out = self.bn(x)
        out = self.conv(self.act(out))
        return torch.cat([x, out], 1)


class DenseBlock_Module(nn.Module):
    """
    |----------------|
    |------|---------|
    block1 -- block2 -- block3
    """
    def __init__(self, nb, in_ch, growth_ch, with_bn, act_type):
        super(DenseBlock_Module, self).__init__()
        blocks = []
        for i in range(nb):
            blocks.append(_DenseBasicBlock(in_ch + i * growth_ch, growth_ch, with_bn, act_type))
        self.dense_layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.dense_layers(x)