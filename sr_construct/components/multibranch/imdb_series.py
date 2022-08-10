from turtle import forward
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_act_layer

# IMDB modified refering to the official code
# https://github.com/Zheng222/IMDN/blob/master/model/block.py
class IMDB_Module(nn.Module):
    """
     -- conv -- split --------------------- cat --- (ESA/CCA etc.) - conv -
                  |__ conv -- split --------|
                                |___ conv --|
        distill_rate: ratio of distilled (direct concat) channel number
        1 / distill_rate: multi-branch level, equals split number + 1, number of concat components
    """
    def __init__(self, n_feat, distill_rate=0.25, act_type='lrelu', attn_module=None) -> None:
        super(IMDB_Module, self).__init__()
        assert 1 / distill_rate % 1 == 0
        self.nf_distill = int(n_feat * distill_rate)
        self.nf_remain = n_feat - self.nf_distill
        self.level = int(1 / distill_rate) # level must be over 2
        if attn_module is not None:
            self.with_attn = True
            self.attn = attn_module
        else:
            self.with_attn = False

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        conv_r_ls = []
        for i in range(self.level - 1):
            if i < self.level - 2:
                conv_r_ls.append(nn.Conv2d(self.nf_remain, n_feat, 3, 1, 1))
            else:
                conv_r_ls.append(nn.Conv2d(self.nf_remain, self.nf_distill, 3, 1, 1))
        self.conv_remains = nn.ModuleList(conv_r_ls)
        self.conv_out = nn.Conv2d(self.nf_distill * self.level, n_feat, 1, 1, 0)
        self.act = get_act_layer(act_type)

    def forward(self, x):
        feat_in = self.conv_in(x)
        out_d0, out_r = torch.split(feat_in, (self.nf_distill, self.nf_remain), dim=1)
        distill_ls = [out_d0]
        for i in range(self.level - 2):
            out_dr = self.act(self.conv_remains[i](out_r))
            out_d, out_r = torch.split(out_dr, (self.nf_distill, self.nf_remain), dim=1)
            distill_ls.append(out_d)
        out_d_last = self.conv_remains[self.level - 2](out_r)
        distill_ls.append(out_d_last)
        fused = torch.cat(distill_ls, dim=1)
        if self.with_attn:
            fused = self.attn(fused)
        out = self.conv_out(fused) + x
        return out

class _SRB(nn.Module):
    """
    SRB used in RFDB
    --- conv3 -- + --
     |___________|
    """
    def __init__(self, nf) -> None:
        super(_SRB, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        out = self.conv(x) + x
        return out

# RFDB modified refering to the official code
# https://github.com/njulj/RFDN
class RFDB_Module(nn.Module):
    """
     -- conv -- conv1 ---------------- cat -- conv -- (ESA/CCA etc.) ->
             |__conv3 --- conv1 --------|
                       |__conv3 --------|
        stage: number of paired conv_d & conv_r, concat number is stage + 1
        attn_module is applied with n_feat (after concat + conv)
    """
    def __init__(self, n_feat, nf_distill, stage,
                 act_type='lrelu', attn_module=None):
        super(RFDB_Module, self).__init__()
        self.nf_distill = nf_distill
        # self.nf_remain = n_feat
        self.stage = stage
        if attn_module is not None:
            self.with_attn = True
            self.attn = attn_module
        else:
            self.with_attn = False

        conv_d_ls = [nn.Conv2d(n_feat, self.nf_distill, 1, 1, 0)]
        conv_r_ls = [_SRB(n_feat)]
        for i in range(1, self.stage):
            conv_d_ls.append(nn.Conv2d(n_feat, self.nf_distill, 1, 1, 0))
            conv_r_ls.append(_SRB(n_feat))
        conv_d_ls.append(nn.Conv2d(n_feat, self.nf_distill, 3, 1, 1))
        self.conv_distill = nn.ModuleList(conv_d_ls)
        self.conv_remains = nn.ModuleList(conv_r_ls)

        self.conv_out = nn.Conv2d(self.nf_distill * (stage + 1), n_feat, 1, 1, 0)
        self.act = get_act_layer(act_type)

    def forward(self, x):
        distill_ls = []
        feat_in = x
        for i in range(self.stage):
            out_d = self.conv_distill[i](x)
            distill_ls.append(out_d)
            x = self.conv_remains[i](x)
        out_d = self.conv_distill[self.stage](x)
        distill_ls.append(out_d)
        fused = torch.cat(distill_ls, dim=1)
        out = self.conv_out(fused)
        if self.with_attn:
            out = self.attn(out)
        return out