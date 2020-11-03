import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import os,sys,inspect,glob,re,time,datetime
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


class CSRNet(Model):
    
    def __init__(self,frontEnd,backEnd,output_layer_arch, weightsFlag=False):
        super(CSRNet, self).__init__()
        # self.seen = 0
        # self.frontEnd_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # self.backEnd_feat  = [512, 512, 512,256,128,64]
        self.frontEnd = layers.construct_net(frontEnd)
        self.backEnd = layers.construct_net(backEnd)
        self.output_layer = layers.construct_net(output_layer_arch) 
            # If the weights are not initialized, use the VGG16 architecture for frontEnd
        if not weightsFlag:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontEnd.state_dict().items())):
                self.frontEnd.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
    
    
    def __init__(self, weightsFlag=False):
        super(CSRNet, self).__init__() 
        self.frontEnd ,self.backEnd ,self.output_layer=self.default_architecture()
        self.frontEnd ,self.backEnd =layers.construct_net(self.frontEnd),layers.construct_net(self.backEnd)
        if not weightsFlag:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontEnd.load_state_dict(mod.features[0:23].state_dict())

    def forward(self,x):
        y = self.frontEnd(x)
        y = self.backEnd(y)
        y = self.output_layer(y)
        y = F.interpolate(y,scale_factor=2, mode='bilinear')
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def default_architecture():
        output_layer=nn.Conv2d(64, 1, kernel_size=1,padding=0)

        return shapes.CSRNET_FRONTEND,shapes.CSRNET_BACKEND,output_layer         

