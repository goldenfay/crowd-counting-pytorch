import torch.nn as nn
import torch
import torch.nn.functional as F
import os,sys,inspect,glob
import numpy as np
    # User's modules
import model
from model import *
import layers
import shapes

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's module from another directory
sys.path.append(os.path.join(parentdir, "bases"))
from utils import BASE_PATH


class SANet(Model):
    
    def __init__(self,rgb_channels=True,uses_batch_norm=True):
        super(SANet, self).__init__()

        in_channels = 3 if rgb_channels else 1

        self.encoder = nn.Sequential(
            shapes.SANetHead(in_channels, 64, uses_batch_norm),
            nn.MaxPool2d(2, 2),
            shapes.SANetCore(64, 128, uses_batch_norm),
            nn.MaxPool2d(2, 2),
            shapes.SANetCore(128, 128, uses_batch_norm),
            nn.MaxPool2d(2, 2),
            shapes.SANetCore(128, 128, uses_batch_norm),
            )

        self.decoder = nn.Sequential(
            layers.BasicConv(128, 64, use_bn=uses_batch_norm, kernel_size=9, padding=4),
            layers.BasicDeconv(64, 64, 2, stride=2, use_bn=uses_batch_norm),
            layers.BasicConv(64, 32, use_bn=uses_batch_norm, kernel_size=7, padding=3),
            layers.BasicDeconv(32, 32, 2, stride=2, use_bn=uses_batch_norm),
            layers.BasicConv(32, 16,  use_bn=uses_batch_norm, kernel_size=5, padding=2),
            layers.BasicDeconv(16, 16, 2, stride=2, use_bn=uses_batch_norm),
            layers.BasicConv(16, 16,  use_bn=uses_batch_norm, kernel_size=3, padding=1),
            layers.BasicConv(16, 1, use_bn=False, kernel_size=1),
            )
        for module in self.modules():
            self._initialize_weights(module)
        
            
    def forward(self,x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out

    def _initialize_weights(self,m):
            if isinstance(m, list):
                for mini_m in m:
                    self._initialize_weights(mini_m)
            else:
                if isinstance(m, nn.Conv2d):    
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m,nn.Module):
                    for mini_m in m.children():
                        self._initialize_weights(mini_m)
                




















    