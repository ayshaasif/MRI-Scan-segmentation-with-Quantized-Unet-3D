# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:46:01 2023

@author: aysha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:05:03 2023
set CUDA_DEVICE_VISIBLE=0,1 & python UNET3D_EagerQuantized_v3_2.py
@author: aysha
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import numpy as np



"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""
import glob
# from torchsummary import summary
from torch.ao.quantization import QuantStub, DeQuantStub


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
#  ----------------------------Defining the custom loss functions ----------------------
#PyTorch
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
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss




# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()
#         self.name = "Dice Loss"
#         self.dice_coeff = 0
        
#     def __str__(self):
#         return self.name
    
#     def forward(self, inputs, targets, smooth=1e-8):
#         # Reshape to (batch_size, classes, depth, height, width)
#         # print(inputs.shape, targets.shape)

#         inputs = inputs.flatten()
#         targets = targets.flatten()
    
#         # print(inputs.shape, targets.shape)
#         intersection = torch.sum(inputs * targets)
#         # print(intersection)
#         self.dice_coeff = (2.0 * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
#         dice_loss = 1.0 - self.dice_coeff
        
#         return dice_loss
    
#     def get_score(self):
#         return self.dice_coeff
    
    
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.name = "Dice Loss"
        self.dice_coef = 0
    def __str__(self):
        return self.name
    
    
    def forward(self, inputs, targets, smooth=1e-8):
        shape = inputs.shape
        class_weight = [0.1,0.3,0.3,0.3]
        print(shape)
        for batch in range(shape[0]):
            for tumor_class in range(shape[4]):
                input_flat = inputs[batch,:,:,:,tumor_class].view(-1)
                target_flat = targets[batch,:,:,:,tumor_class].view(-1)
                intersection = (input_flat * target_flat ).sum()
                W = class_weight[tumor_class]
                self.dice_coef += W*(1 - ((2*intersection +  smooth)/(input_flat.sum() +  target_flat.sum()+ smooth)))
        return -self.dice_coef

            
    def get_score(self):
        return self.dice_coef


# ---------------------------------------loss functions defined --------------------------

# dc = DiceLoss()
# dl = dc(torch.Tensor([1,0,10]),torch.Tensor([1,0,10]))
# print(dc.get_score())
# print(dl)

batch_size = 10  # Replace with your batch size
preds = torch.randn(batch_size, 128, 128, 128, 4)  # Replace with your predicted tensor
target = torch.randint(0, 2, (batch_size, 128, 128, 128, 4)) # Replace with your target tensor

# loss = DiceLoss()
# dc_loss = loss(predicted, predicted)
# print(loss.get_score())
# print("Dice Loss:", dc_loss.item())


# from torch import tensor
from torchmetrics.classification import Dice
dice = Dice(average='micro')
print(dice(target, target))








