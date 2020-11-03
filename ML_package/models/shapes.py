import torch
import torch.nn as nn
import torch.nn.functional as F

import layers

CSRNET_FRONTEND = [

    ('C2D', {
        'in_channels': 3,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 64,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 64,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    })


]

CSRNET_BACKEND = [
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 512,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 512,
        'out_channels': 256,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 256,
        'out_channels': 128,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 128,
        'out_channels': 64,
        'ks': 3,
        'stride': 1,
        'padding': 2,
        'dilation': 2
    }),
    ('R', {
        'inplace': True
    })
]

CCNN_BACKEND=[
    ('C2D', {
        'in_channels': 40,
        'out_channels': 40,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 40,
        'out_channels': 60,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 60,
        'out_channels': 40,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('M', {
        'ks': 2,
        'stride': 2,

    }),
    ('C2D', {
        'in_channels': 40,
        'out_channels': 20,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    }),
    ('C2D', {
        'in_channels': 20,
        'out_channels': 10,
        'ks': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1
    }),
    ('R', {
        'inplace': True
    })
]


class SANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SANetHead, self).__init__()
        branch_out = out_channels // 4
        self.branch_1x1 = layers.BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch_3x3 = layers.BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.branch_5x5 = layers.BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2)
        self.branch_7x7 = layers.BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3)
    
    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        branch_7x7 = self.branch_7x7(x)
        out = torch.cat([branch_1x1, branch_3x3, branch_5x5, branch_7x7], 1)
        return out


class SANetCore(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SANetCore, self).__init__()
        branch_out = out_channels // 4
        self.branch_1x1 = layers.BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch_3x3 = nn.Sequential(
                        layers.BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        layers.BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1),
                        )
        self.branch_5x5 = nn.Sequential(
                        layers.BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        layers.BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2),
                        )
        self.branch_7x7 = nn.Sequential(
                        layers.BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        layers.BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3),
                        )
    
    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        branch_7x7 = self.branch_7x7(x)
        out = torch.cat([branch_1x1, branch_3x3, branch_5x5, branch_7x7], 1)
        return out
