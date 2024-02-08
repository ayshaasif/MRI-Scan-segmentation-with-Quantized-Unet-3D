# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:11:51 2023

@author: aysha
"""

#-----------------------------------------------------------------
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from BraTSDatasetClassFile import BraTSDataset
import torch
from torchvision import transforms
import numpy as np


# BASE_DIR = "../../../data/BratsDataset"
#/data/BratsDataset

BASE_DIR = "../BratsDataset"

IMG_DIR = "images"
MASK_DIR = "masks"

train_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,transform = torch.Tensor)
test_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,test=True,transform = torch.Tensor)
num_train_examples = 100 # change the number of samples
sample_ds = Subset(train_mri_dataset, np.arange(num_train_examples))


train_batch_size = 4
test_batch_size = 4



pin_memory = True

train_data_loader = DataLoader(train_mri_dataset, batch_size=train_batch_size,shuffle=True,
                               num_workers=0, pin_memory=pin_memory)


test_data_loader = DataLoader(test_mri_dataset, batch_size=test_batch_size,
	num_workers=0, pin_memory=pin_memory,shuffle=True) 


sample_ds_loader = DataLoader(sample_ds, batch_size=2,
	num_workers=0, pin_memory=True,shuffle=True)

# train_mri_dataset.__getitem__(slice(160,274))
# batch = next(iter(sample_ds_loader))
# print(len(batch))