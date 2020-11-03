
import torch
from torch import nn
class TrainParams:
    
    def __init__(self,device,module:nn.Module,lr,momentum,maxEpochs,criterionMethode,optimizationMethod,**kargs):
        super().__init__()
        self.device=device
        self.lr=lr
        self.momentum=momentum
        self.maxEpochs=maxEpochs
        if criterionMethode.lower()=="MSELoss".lower():
            self.criterion=nn.MSELoss(size_average=False).to(device)
        elif criterionMethode.lower()=="CrossEntropyLoss".lower():
            self.criterion=nn.CrossEntropyLoss(size_average=False).to(device)
            
        elif criterionMethode.lower()=="L1Loss".lower():
            self.criterion=nn.L1Loss(size_average=False).to(device)
        elif criterionMethode.lower()=="L2Loss".lower():
            self.criterion=nn.MSELoss(size_average=False).to(device)
        elif criterionMethode.lower()=="CrossEntropy".lower():
            self.criterion=nn.CrossEntropyLoss(size_average=False).to(device)
        else: raise TypeError("Invalide argument\n\t 'Criterion' Value is invalide ")

        if optimizationMethod.lower()=="SGD".lower():
            self.optimizer=torch.optim.SGD(module.parameters(), lr=self.lr,
                                momentum=self.momentum)
        elif optimizationMethod.lower()=="Adam".lower():
            self.optimizer=torch.optim.Adam(module.parameters(), lr=self.lr)
            
        elif optimizationMethod.lower()=="RMSprop".lower():
            self.optimizer=torch.optim.RMSprop(module.parameters(), lr=self.lr,
                                momentum=self.momentum)
        elif optimizationMethod.lower()=="LBFGS".lower():
            self.optimizer=torch.optim.LBFGS(module.parameters(), lr=self.lr)
        else: raise TypeError("Invalide argument\n\t 'Criterion' Value is invalide ")
        



    @staticmethod
    def defaultTrainParams():
    
        return None