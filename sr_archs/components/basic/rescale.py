from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_basic import get_act_layer

# UP SCALE
class PixelShuffle_Module(nn.Module):
    """
    conv + pixel-shuffle
    """
    def __init__(self, n_feat, out_nc, scale) -> None:
        super(PixelShuffle_Module, self).__init__()
        nf_out = n_feat * (scale ** 2)
        self.conv = nn.Conv2d(n_feat, nf_out, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        hr_feat = self.conv(x)
        out = self.ps(hr_feat)
        return out


class DeconvUp_Module(nn.Module):
    def __init__(self, n_feat, out_nc, scale,
                last_conv=False, nf=None, ksize=None):
        super(DeconvUp_Module, self).__init__()
        self.last_conv = last_conv
        if not last_conv:
            self.up = nn.ConvTranspose2d(n_feat, out_nc, scale, stride=scale)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(n_feat, nf, scale, stride=scale),
                nn.Conv2d(nf, out_nc, ksize, 1, ksize // 2)
            )

    def forward(self, x):
        hr = self.up(x)
        return hr


class InterpUp_Module(nn.Module):
    def __init__(self, scale, interp='nearest',
                last_conv=False, n_feat=None, nf=None, ksize=None, act_type='relu'):
        super(InterpUp_Module, self).__init__()
        self.interp = interp
        self.scale = scale
        self.last_conv = last_conv
        if last_conv:
            if act_type is not None:
                act_layer = get_act_layer(act_type, nf)
                self.conv = nn.Sequential(
                    nn.Conv2d(n_feat, nf, ksize, 1, ksize // 2),
                    act_layer
                )
            else:
                self.conv = nn.Conv2d(n_feat, nf, ksize, 1, ksize // 2)
        
    def forward(self, x):
        x_large = F.interpolate(x, scale_factor=self.scale, mode=self.interp)
        if self.last_conv:
            x_large = self.conv(x_large)
        return x_large


# DOWN SCALE
class PoolDown_Module(nn.Module):
    pass

class PixelUnshuffle_Module(nn.Module):
    pass