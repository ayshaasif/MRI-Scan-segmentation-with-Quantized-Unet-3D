
import torch
import torch.nn as nn       

fct_ALPHA = 0.5
fct_BETA = 0.5
fct_GAMMA = 1


    

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.name = "Focal Tversky Loss"
    
    def __str__(self):
        return self.name
    
    
    def forward(self, inputs, targets, smooth=1, alpha=fct_ALPHA, beta=fct_BETA, gamma=fct_GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs,dim=1)       
        
        #flatten label and prediction tensors
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
    