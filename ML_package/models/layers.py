import torch
from torch import nn
import torch.nn.functional as F



def conv2D_layer(params:dict):
    return nn.Conv2d(params['in_channels'],
                    params['out_channels'],
                    kernel_size=params['ks'],
                    stride=params['stride'],
                    padding=params['padding'],
                    dilation=params['dilation']
                    )
def relu_layer(params:dict):
    return nn.ReLU(inplace=params['inplace'])

def maxpool_layer(params:dict):
    return nn.MaxPool2d(params['ks'],stride=params['stride'])

def fc_layer(params:dict):
    return nn.Linear(params['in_size'],params['out_size'])

def insnorm_layer(params:dict):
    return nn.InstanceNorm2d(params['out_channels'],affine=True)    


def construct_net(schema:list,weight_flag=False):
    arch=[]
    for layer_key,params in schema:
        if layer_key=='M':
            arch.append(maxpool_layer(params))
        elif layer_key=='C2D':
            arch.append(conv2D_layer(params))
        elif layer_key=='R':
            arch.append(relu_layer(params))
        elif layer_key=='BR':
            arch.append(nn.BatchNorm2d(params['out_channels']))    
            arch.append(relu_layer(params))    
        elif layer_key=='FC':
            arch.append(fc_layer(params))
        elif layer_key=='IN':
            arch.append(insnorm_layer(params))       

    return nn.Sequential(*arch)        


    # #####################################Classes#####################################################"


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.batch_norm = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.batch_norm, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.batch_norm, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.batch_norm = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.batch_norm)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.batch_norm else None

    def forward(self, x):
        x = self.tconv(x)
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class SumPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(SumPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        if type(kernel_size) is not int:
            self.area = kernel_size[0] * kernel_size[1]
        else:
            self.area = kernel_size * self.kernel_size
    
    def forward(self, x):
        return self.avgpool(x) * self.area        
