import torch
import torch.nn as nn


class SE_Module(nn.Module):
    """
    SE module for channel attention
    [Block] - GlobalPool - FC - ReLU - FC - Sigmoid - x -
        |_____________________________________________|

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(SE_Module, self).__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attn(x)
        return x * y


class CCA_Module(nn.Module):
    """
    Contrast-aware Channel Attention module
    refer to: https://github.com/Zheng222/IMDN/blob/master/model/block.py

    [Block] -- ChannelStd -|
        |_____ GlobalPool -+- FC - ReLU - FC - Sigmoid - x -
        |________________________________________________|

        num_feat (int): 
        squeeze_factor (int): 
    """
    def __init__(self, channel, reduction=16):
        super(CCA_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def stdv_channels(self, featmap):
        """
        featmap (torch.tensor): [b, c, h, w]
        """
        assert(featmap.dim() == 4)
        channel_mean = featmap.mean(3, keepdim=True).mean(2, keepdim=True)
        channel_variance = (featmap - channel_mean).pow(2).mean(3, keepdim=True).mean(2, keepdim=True)
        return channel_variance.pow(0.5)

    def forward(self, x):
        y = self.stdv_channels(x) + self.avg_pool(x)
        y = self.attn(y)
        return x * y