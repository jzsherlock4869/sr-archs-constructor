import torch
import torch.nn as nn

def get_act_layer(act_type='relu', n_feat=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'prelu':
        return nn.PReLU(n_feat)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.05, inplace=True)
    else:
        raise NotImplementedError
