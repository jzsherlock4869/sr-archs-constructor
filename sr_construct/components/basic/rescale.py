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
    """
        deconv[n_feat -> out_nc/nf] -- (conv[nf -> out_nc]) ->
    """
    def __init__(self, n_feat, out_nc, scale,
                last_conv=False, nf=None, ksize=None, act_type='relu'):
        super(DeconvUp_Module, self).__init__()
        self.last_conv = last_conv
        if not last_conv:
            self.up = nn.ConvTranspose2d(n_feat, out_nc, scale, stride=scale)
        else:
            if act_type is not None:
                act_layer = get_act_layer(act_type, out_nc)
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(n_feat, nf, scale, stride=scale),
                    nn.Conv2d(nf, out_nc, ksize, 1, ksize // 2),
                    act_layer
                )
            else:
                self.up = nn.Sequential(
                    nn.ConvTranspose2d(n_feat, nf, scale, stride=scale),
                    nn.Conv2d(nf, out_nc, ksize, 1, ksize // 2)
                )

    def forward(self, x):
        hr = self.up(x)
        return hr


class InterpUp_Module(nn.Module):
    """
        interp -- (conv[n_feat -> nf]) ->
    """
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
    """
        -- pool -->
    """
    def __init__(self, scale, pool_type='avg'):
        super(PoolDown_Module, self).__init__()
        self.scale = scale
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(scale)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(scale)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        if x.size(2) % self.scale != 0 \
            or x.size(3) % self.scale != 0:
            print(f'[Warning][PoolDown_Module] {x.size()} not divisible by scale {self.scale}')
        output = self.pool(x)
        return output


class PixelUnshuffle_Module(nn.Module):
    """
    pixel-unshuffle[n_feat->n_feat*(s^2)] -- (conv[n_feat*(s^2)->nf]) -->
    """
    def __init__(self, n_feat, scale,
                last_conv=False, nf=None, ksize=None, act_type='relu'):
        super(PixelShuffle_Module, self).__init__()
        self.last_conv = last_conv
        self.pus = nn.PixelUnshuffle(scale)

        if last_conv:
            self.pus_out_nc = n_feat * (scale ** 2)
            if act_type is not None:
                act_layer = get_act_layer(act_type, nf)
                self.conv = nn.Sequential(
                    nn.Conv2d(self.pus_out_nc, nf, ksize, 1, ksize // 2),
                    act_layer
                )
            else:
                self.conv = nn.Conv2d(self.pus_out_nc, nf, ksize, 1, ksize // 2)
    
    def forward(self, x):
        output = self.pus(x)
        if self.last_conv:
            output = self.conv(output)
        return output