
import torch
import torch.nn as nn


    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.name = "IOU loss"
    
    def __str__(self):
        return self.name
    
    def forward(self, inputs, targets, smooth=1):
                
        #flatten label and prediction tensors
        inputs = torch.softmax(inputs, dim=1)
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    