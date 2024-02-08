# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:27:58 2023

@author: aysha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from BraTSDatasetClassFile import BraTSDataset
from UNet3D import UNet3D
import torchmetrics
from torchmetrics.classification import Dice
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
import json
from torch.utils.data import Subset, DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib
import datetime



def logParams(fname=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), lr=0.001, optimizer="Adam",epochs=50, TrainingLoss=None, TrainDiceScore=None, loss="weighted cross entropy + 0.01*diceloss"):
    try:
        with open("log.txt", "a") as f:
            logRecord = f'model checkpoint name: {fname},   learning rate: {lr}, optimizer:{optimizer}, epochs:{epochs}, TrainLoss: {TrainingLoss}, TrainDiceScore:{TrainDiceScore} , loss:{loss}\n'
            f.write(logRecord)
        return "--------------Log Updated ---------------"
    except:
        print("File not found")
        
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M_{fname}.pt'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

matplotlib.use('Agg')

BASE_DIR = "../BratsDataset"

IMG_DIR = "images"
MASK_DIR = "masks"

train_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,transform = torch.Tensor)
test_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,test=True,transform = torch.Tensor)
num_train_examples = 170 # change the number of samples
sample_ds = Subset(train_mri_dataset, np.arange(num_train_examples))
    
 
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # setting the device id , change to "rank" if u wanna use all the gpus

lambda1 = lambda epoch: epoch//5 if epoch< 10 else epoch // 10

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,        
    ) -> None:
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler_rlp = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer = optimizer,
                            max_lr = 3e-2,
                            epochs = 100,
                            steps_per_epoch = len(sample_ds) // 4,
                            verbose=True
                        )
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.history = {
                        "Train_loss":[],
                        "Train_dice_score":[]
                        }

        
        
    def _run_batch(self, source, targets):
        # print("self.gpuid: ",self.gpu_id)
        self.optimizer.zero_grad()
        # print(f"source min: {torch.min(source)} and max: {torch.max(source)}")
        output = self.model(source)
        # print(f" output : {output.shape}")
        s_output = torch.nn.Softmax(dim=1)(output)#(1,4,128,128,128)
        # print(f"targets min: {torch.min(targets)} and max: {torch.max(targets)}")
        # print(f"pred min: {torch.min(s_output)} and max: {torch.max(s_output)}")
        target_indices = torch.argmax(targets, dim=1) #(1,128,128,128)
        # print(f"target_indices: {target_indices.shape}")
        # print(f"soutput: {s_output.shape}, targets: {targets.int().shape[1]}")
        output_indices =  torch.argmax(s_output, dim=1)
        # print(f"output indices: {output_indices.shape}, targets: {target_indices.shape}")

        self.dc_score  = torchmetrics.functional.dice(output_indices,target_indices,
                                                      num_classes=4, average='macro')
        diceloss = 1 - self.dc_score
        diceloss.requires_grad = True

        # self.dc_score = diceLoss.getScore()
        weight = torch.Tensor([0.00414074, 0.53114366, 0.12849232, 0.33622329]).to(self.gpu_id)
        ce_loss = F.cross_entropy(output, target_indices, weight=weight)
        self.loss = ce_loss + 0.01*diceloss
        print(f"dice score: {self.dc_score}, ce_loss:{ce_loss}, loss:{self.loss}")

        self.loss.backward()
        self.optimizer.step()
        self.tot_score = self.dc_score.item()
        self.tot_loss = self.loss.item()
        # print(f"tot score: {self.tot_score} loss: {self.tot_loss}")
        self.scheduler.step()




    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        train_losses = []
        train_dc = []
        for source, targets in self.train_data:
            source = torch.permute(source, [0,4,1,2,3]).to(self.gpu_id)
            targets = torch.permute(targets, [0,4,1,2,3]).to(self.gpu_id)
            self._run_batch(source, targets)
            train_losses.append(self.tot_loss)
            train_dc.append(self.tot_score)

        train_loss_epoch = sum(train_losses) / len(train_losses)
        train_dice_epoch = sum(train_dc) / len(train_dc)
        print(f"train loss epoch: {train_loss_epoch},dice epoch: {train_dice_epoch}")

        
        self.history['Train_loss'].append(train_loss_epoch)
        self.history['Train_dice_score'].append(train_dice_epoch)
        print("\n")
        self.scheduler_rlp.step(self.loss,epoch=10)


    def _save_checkpoint(self, epoch):
        print(f"train epoch: {self.history['Train_dice_score'][-1]}")
        if self.history['Train_dice_score'][-1] > 0.6:
            ckp = self.model.module.state_dict()
            PATH = timeStamped("checkpoint")
            torch.save(ckp, PATH)
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}| dice score : {self.history['Train_dice_score'][-1]} | loss: {self.history['Train_loss'][-1]}")
        else:
            print("model not good enough")
            
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
    def save_json(self):
        try:
            with open("{}.json".format("checkpoint"), "w") as f:
                print(f" Train loss: {self.history['Train_loss'][0]} Dice score : {self.history['Train_dice_score'][0]}")

                json.dump(self.history,f)
        except :
            print("Error saving the History....")
                


def load_train_objs():
    train_set = sample_ds  # load your dataset
    model = UNet3D(3, 4)  # load your model
    model.load_state_dict(torch.load('2023-12-20-00-00_checkpoint.pt'))
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    trainer.save_json()
    logParams( lr=0.01, optimizer="Adam",epochs=200,
              TrainingLoss=trainer.history['Train_loss'][-1], TrainDiceScore=trainer.history['Train_dice_score'][-1])

    destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
   
    
    world_size = torch.cuda.device_count()
    # world_size = 1

    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
