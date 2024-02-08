# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 01:35:41 2023

@author: aysha
"""

from UNet3D import UNet3D
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp
from HelperFunctions import  train, evaluate_model
from DataLoader import sample_ds_loader,test_data_loader, train_data_loader, train_mri_dataset, sample_ds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


if __name__ == '__main__':    

    #----------------------------------------Testing--------------------------------------------------------------------
    
    model = UNet3D(3, 4)
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0,2])
        print("parallel  model created......")
        
        
        
        
    file_name = "newfile_29102023"
    print("train_mri_dataset: {},train_data_loader:  {}".format(len(train_mri_dataset), len(train_data_loader)))
    print("sample_mri_dataset: {},sample_data_loader:  {}".format(len(sample_ds), len(sample_ds_loader)))
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            train(model,train_data_loader,sample_ds_loader,10,file_name=file_name,de=torch.device("cuda"))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("Training complete")
    # evaluate_model(model, test_data_loader, device=torch.device("cuda"), file_name=file_name)
    
