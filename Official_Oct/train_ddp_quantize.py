# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:29:21 2023

@author: aysha
"""

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
from torch.quantization.observer import MinMaxObserver,MovingAverageMinMaxObserver,PerChannelMinMaxObserver
from tqdm import tqdm
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
        
def timeStamped(fname, fmt='%Y-%m-%d-%H-_{fname}.pt'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

matplotlib.use('Agg')

BASE_DIR = "../BratsDataset"

IMG_DIR = "images"
MASK_DIR = "masks"

train_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,transform = torch.Tensor)
test_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,test=True,transform = torch.Tensor)
num_train_examples = 15 # change the number of samples
sample_ds = Subset(train_mri_dataset, np.arange(num_train_examples))
    
 
# def ddp_setup(rank, world_size):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     init_process_group(backend="gloo", rank=rank, world_size=world_size)

# lambda1 = lambda epoch: epoch//5 if epoch< 10 else epoch // 10

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,        
    ) -> None:
        
        # self.gpu_id = gpu_id
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

        self.save_every = save_every
        # self.model = DDP(model)
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
                                                       ignore_index=0,
                                                      num_classes=4, average='macro')
        diceloss = 1 - self.dc_score
        diceloss.requires_grad = True

        # self.dc_score = diceLoss.getScore()
        weight = torch.Tensor([0.00414074, 0.53114366, 0.12849232, 0.33622329])
        ce_loss = F.cross_entropy(output, target_indices, weight=weight)
        self.loss = ce_loss + 0.01*diceloss
        print(f"dice score: {self.dc_score}, ce_loss:{ce_loss}, loss:{self.loss}")

        self.loss.backward()
        self.optimizer.step()
        self.tot_score = self.dc_score.item()
        self.tot_loss = self.loss.item()
        # print(f"tot score: {self.tot_score} loss: {self.tot_loss}")




    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f" Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # self.train_data.sampler.set_epoch(epoch)
        train_losses = []
        train_dc = []
        for source, targets in tqdm(self.train_data):
            source = torch.permute(source, [0,4,1,2,3])
            targets = torch.permute(targets, [0,4,1,2,3])
            self._run_batch(source, targets)
            train_losses.append(self.tot_loss)
            train_dc.append(self.tot_score)
        self.scheduler.step(self.loss)

        train_loss_epoch = sum(train_losses) / len(train_losses)
        train_dice_epoch = sum(train_dc) / len(train_dc)
        print(f"train loss epoch: {train_loss_epoch},dice epoch: {train_dice_epoch}")

        
        self.history['Train_loss'].append(train_loss_epoch)
        self.history['Train_dice_score'].append(train_dice_epoch)
        print("\n")


    def _save_checkpoint(self, epoch):
        print(f"train epoch: {self.history['Train_dice_score'][-1]}")
        self.model.to("cpu")
        self.model = torch.quantization.convert(self.model,inplace=True)  # load your model
        print(self.model)

        # if self.history['Train_dice_score'][-1] > 0.6:
        ckp = self.model.module.state_dict()
        PATH = timeStamped("checkpoint")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}| dice score : {self.history['Train_dice_score'][-1]} | loss: {self.history['Train_loss'][-1]}")
        # else:
        #     print("model not good enough")
            
    def train(self, max_epochs: int):
        for epoch in tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            print(self.model)
            if epoch % self.save_every == 0:
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
    qat_model = UNet3D(3, 4, quantize=True)
    qat_model.load_state_dict(torch.load('2023-12-02-12-_checkpoint.pt'), strict=False)
    qat_model.to("cpu")
    qat_model.eval()
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig("onednn") #backend
    qat_model.qconfig = torch.quantization.QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_affine),
                                                   weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_channel_symmetric))
    
    qat_model.s_block1.upconv1.qconfig = None
    qat_model.s_block2.upconv1.qconfig = None
    qat_model.s_block3.upconv1.qconfig = None    
    modules_to_fuse = [['a_block1.conv1','a_block1.bn1','a_block1.relu1'],
                        ['a_block1.conv2','a_block1.bn2','a_block1.relu2'],
                        ['a_block2.conv1','a_block2.bn1','a_block2.relu1'],
                        ['a_block2.conv2','a_block2.bn2','a_block2.relu2'],
                        ['a_block3.conv1','a_block3.bn1','a_block3.relu1'],
                        ['a_block3.conv2','a_block3.bn2','a_block3.relu2'],
                        
                        ['s_block1.conv1','s_block1.bn1','s_block1.relu1'],
                        ['s_block1.conv2','s_block1.bn2','s_block1.relu2'],
                        ['s_block2.conv1','s_block2.bn1','s_block2.relu1'],
                        ['s_block2.conv2','s_block2.bn2','s_block2.relu2'],
                        ['s_block3.conv1','s_block3.bn1','s_block3.relu1'],
                        ['s_block3.conv2','s_block3.bn2','s_block3.relu2'],
                        ]
    torch.quantization.fuse_modules(qat_model,modules_to_fuse, inplace=True)
    
    qat_model.train()
    qat_model= torch.quantization.prepare_qat(qat_model, inplace=True)
    # qat_model.to("cuda")
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=0.001)
    print("model prepared")
    return train_set, qat_model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        # sampler=RandomSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int):
    # ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer,  save_every)
    print('training started ..... ')
    trainer.train(total_epochs)
    print("Training done")
    trainer.save_json()
    logParams( lr=0.001, optimizer="Adam",epochs=10,
              TrainingLoss=trainer.history['Train_loss'][-1], TrainDiceScore=trainer.history['Train_dice_score'][-1])

    # destroy_process_group()



if __name__ == "__main__":
    print("backends:",torch.backends.quantized.supported_engines)
    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()
   
    
    # world_size = torch.cuda.device_count()
    # world_size = 2
    main(1,2,2)
    # mp.spawn(main, args=(world_size, 1, 2, 2), nprocs=world_size)