import torch
import torch.nn as nn  
    
    
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.name = "Focal loss"
    
    def __str__(self):
        return self.name
    
    
    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):      
        
        #flatten label and prediction tensors
        inputs = torch.softmax(inputs, dim=1)

        inputs = inputs.flatten()
        targets = targets.flatten()
        
        #first compute binary cross-entropy 
        BCE = nn.BinaryCrossEntropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    