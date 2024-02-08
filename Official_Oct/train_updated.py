# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:28:30 2023

@author: aysha
"""
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from BraTSDatasetClassFile import BraTSDataset
from DataLoader import train_data_loader, test_data_loader, sample_ds_loader, train_mri_dataset, test_mri_dataset
from UNet3D import UNet3D
from HelperFunctions import train_loop, test_loop, train_loop_dataset, test_loop_dataset
from tqdm import tqdm 

if __name__ =='__main__':
    model = UNet3D(3,4)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate=0.01
    optimizer = SGD(model.parameters(), lr=learning_rate)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 5
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0,2])
        print("parallel  model created......")
        
        
    file_name = "newfile_01112023_v2"
    print("train_data_loader:  {}".format(len(train_data_loader)))
    print("test_data_loader:  {}".format(len(test_data_loader)))
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(sample_ds_loader, model, loss_fn, optimizer, device, file_name)
        test_loop(test_data_loader, model, loss_fn,device)
        # train_loop_dataset(train_mri_dataset, model, loss_fn, optimizer, device, file_name)
        # test_loop_dataset(test_mri_dataset, model, loss_fn, device)
    print("Done!")