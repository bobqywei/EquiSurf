import torch.nn as nn

from . srresnet import Upsampler

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR(nn.Module):
    def __init__(self, scale, n_resblocks=16, n_feats=256, res_scale=0.1, kernel_size=3):
        super(EDSR, self).__init__()

        # define head module
        m_head = [nn.Conv2d(3, n_feats, kernel_size, padding=(kernel_size//2))]

        # define body module
        m_body = [
            ResBlock(
                n_feats, kernel_size, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats, act=False, bias=True),
            nn.Conv2d(n_feats, 3, kernel_size, padding=(kernel_size//2))
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x
