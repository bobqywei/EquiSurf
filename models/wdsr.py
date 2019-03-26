import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Block(nn.Module):
    def __init__(
        self, n_feats, kernel_size, block_feats, wn, res_scale=1, act=nn.ReLU(True)):
        super(Block, self).__init__()
        self.res_scale = res_scale

        body = [
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size//2)),
            act,
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size//2))
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class WDSR_A(nn.Module):
    def __init__(self, scale, n_resblocks=16, n_feats=128, block_feats=512, kernel_size=3, res_scale=1.0):
        super(WDSR_A, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)

        # define head module
        head = [wn(nn.Conv2d(3, n_feats, 3, padding=1))]

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(Block(n_feats, kernel_size, block_feats, wn=wn, res_scale=res_scale))

        # define tail module
        out_feats = scale*scale*3
        tail = [wn(nn.Conv2d(n_feats, out_feats, 3, padding=1)), nn.PixelShuffle(scale)]
        skip = [wn(nn.Conv2d(3, out_feats, 5, padding=2)), nn.PixelShuffle(scale)]

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x
