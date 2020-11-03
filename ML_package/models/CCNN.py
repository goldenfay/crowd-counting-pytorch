import torch
from torch import nn
import os
    # User's modules
import model
from model import *
import layers
import shapes
import multiprocessing


class CCNN(Model):
    
    def __init__(self,weightsFlag=False):
        super(CCNN,self).__init__()
        self.build(weightsFlag)


    def build(self,weightsFlag):
        print("####### Building Net architecture...")
        self.parallel_layer=[
            nn.Sequential(nn.Conv2d(3,10,9,padding=1),nn.MaxPool2d(2,padding=1)),
            nn.Sequential(nn.Conv2d(3,14,7),nn.MaxPool2d(2,padding=1)),
            nn.Sequential(nn.Conv2d(3,16,5),nn.MaxPool2d(2))
        ]

        self.backend=layers.construct_net(shapes.CCNN_BACKEND)

        self.output_layer=nn.Conv2d(10,1,1)

        if not weightsFlag:
            self._initialize_weights()
              

    def forward(self,img_tensor):
        # if len(img_tensor.shape)==3: 
        #     import numpy as np
        #     img_tensor=torch.tensor(img_tensor[np.newaxis,:,:,:],dtype=torch.float)

        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # jobs=[multiprocessing.Process(target=self.parallel_process, 
        #                                     args=(i,img_tensor,return_dict)) for i in range(len(self.parallel_layer))]
        # for j in jobs:
        #     j.start()

        # for j in jobs:
        #     j.join()  
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for index in range(len(self.parallel_layer)):
            self.parallel_layer[index].to(device)
        a,b,c=self.parallel_layer[0](img_tensor),self.parallel_layer[1](img_tensor),self.parallel_layer[2](img_tensor)    
        x=torch.cat((a,b,c),1)
        x=self.backend.to(device)(x)
        x=self.output_layer.to(device)(x)
        return x

    def _initialize_weights(self):
        '''
            Initialize the Net weights if those laters are not set on Net définition.
        '''
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0) 

    def parallel_process(self,index,tensor,queue):
        x=self.parallel_layer[index](tensor)
        queue[str(index)]=x

