# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:22:33 2023

@author: aysha
"""
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from DataLoader import train_mri_dataset, train_data_loader

true_labels = []
for x, y in train_mri_dataset:
    y = np.argmax(y,axis=3)
    true_labels.append(y)
    
    
print("true labels generated")
    
unique_labels = [0,1,2,3]
class_weights = compute_class_weight('balanced', classes=unique_labels, y=true_labels)
print(class_weights)