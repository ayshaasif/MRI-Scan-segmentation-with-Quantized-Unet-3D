# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:57:56 2023

@author: aysha
"""

import torch
from BraTSDatasetClassFile import BraTSDataset
from DataLoader import train_data_loader, test_data_loader, sample_ds_loader, train_mri_dataset, test_mri_dataset
from UNet3D import UNet3D
from HelperFunctions import train_loop, test_loop, train_loop_dataset, test_loop_dataset
from tqdm import tqdm 
from DataLoader import train_mri_dataset, test_mri_dataset
from torch.utils.data import DataLoader
import os
import sys
import tempfile
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

if __name__ =='__main__':
    
    train_data_loader = DataLoader(
        train_mri_dataset,
        32,
        shuffle=False,
        sampler=DistributedSampler(train_mri_dataset)
        )
    test_data_loader = DataLoader(
        test_mri_dataset,
        32,
        shuffle=False,
        sampler=DistributedSampler(test_mri_dataset)
        )
    
    model = UNet3D(3,4)
    model = DDP(model, device_ids=[0])
    loss_fn = nn.CrossEntropyLoss()
    learning_rate=0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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