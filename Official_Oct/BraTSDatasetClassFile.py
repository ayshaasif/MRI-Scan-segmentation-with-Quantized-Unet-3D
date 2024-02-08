# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:50:10 2023

@author: aysha
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import numpy as np
import glob

from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, root_dir,image_dir, mask_dir, train_test_split=0.2,test=False,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.test = test
        self.train_test_split = train_test_split
        
    def __len__(self):
        
        if not self.test: 
            len_img = int(len(glob.glob(os.path.join(self.root_dir,self.image_dir, "*.npy")))*(1 - self.train_test_split))
            len_mask = int(len(glob.glob(os.path.join(self.root_dir,self.mask_dir, "*.npy")))*(1 - self.train_test_split))
        else:
            len_img = int(len(glob.glob(os.path.join(self.root_dir,self.image_dir, "*.npy")))* self.train_test_split)
            len_mask = int(len(glob.glob(os.path.join(self.root_dir,self.mask_dir, "*.npy")))* self.train_test_split)
            
        
        assert len_img == len_mask
        return len_img
    
    def __getitem__(self, index):
        len_mask = self.__len__()
        
        if not self.test:
            img = sorted(glob.glob(os.path.join(self.root_dir,self.image_dir, "*.npy")))[index]
            mask = sorted(glob.glob(os.path.join(self.root_dir,self.mask_dir, "*.npy")))[index]
        else:
            img = sorted(glob.glob(os.path.join(self.root_dir,self.image_dir, "*.npy")))[len_mask:][index]
            mask = sorted(glob.glob(os.path.join(self.root_dir,self.mask_dir, "*.npy")))[len_mask:][index]
        img = np.load(img)
        mask = np.load(mask)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            
            
        return img,mask
    