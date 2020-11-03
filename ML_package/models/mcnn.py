import torch
from torch import nn
import os
    # User's modules
import model
from model import *



class MCNN(Model):
    
    def __init__(self,weightsFlag=False):
        super(MCNN,self).__init__()
        self.build(weightsFlag)


    def build(self,weightsFlag):
        print("####### Building Net architecture...")
        
        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0))

        if not weightsFlag:
            self._initialize_weights()
              

    def forward(self,img_tensor):
        if len(img_tensor.shape)==3: 
            import numpy as np
            img_tensor=torch.tensor(img_tensor[np.newaxis,:,:,:],dtype=torch.float)
              
        branch1=self.branch1(img_tensor)
        branch2=self.branch2(img_tensor)
        branch3=self.branch3(img_tensor)
        x=torch.cat((branch1,branch2,branch3),1)
        x=self.fuse(x)
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
                  

if __name__=="__main__":
    # import matplotlib.pyplot as plt
    # # img=torch.rand((3,800,1200),dtype=torch.float)
    # img_rootPath="C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\train_data\\images"

    # img=plt.imread(os.path.join(img_rootPath,"IMG_10.jpg"))
    # img=torch.from_numpy(img.transpose(2,0,1))
    # mcnn=MCNN()
    # out_dmap=mcnn(img)
    # # print(out_dmap.shape)
    # # plt.imshow(  img  )
    # # plt.imshow(  out_dmap.permute()  )
    # # plt.show()
    # x=vis.Visdom()
    # # x.images(img,1,10)
    # x.image(win=5,img=img,opts=dict(title='img'))
    # x.image(win=5,img=out_dmap/(out_dmap.max())*255,opts=dict(title='est_dmap('))

        #    code for getting all methods of a class and then get the doc of every method
    import inspect

    l=dict((k,v) for k,v in MCNN.__dict__.items() if not (k.startswith('__') and k.endswith('__')))
    print(l['build'].__doc__)
    