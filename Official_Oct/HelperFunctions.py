# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:13:06 2023

@author: aysha
"""
import time
from UNet3D import UNet3D
import torch
import os
import gc
from LossFile import  DiceLoss
import json

import numpy as np
from torch.optim import Adam, SGD
import torch.nn as nn
from tqdm import tqdm_notebook, tqdm
from DataLoader import test_batch_size, train_batch_size

def load_model(model_file,de):
    model = UNet3D(in_channels=3, num_classes=4)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to(de)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    

 # ------------------------------------
        
    



def train(unet,train_dsl,val_ds,number_of_epochs,file_name,de="cuda"):
    print("cached data:{} memory allocated: {} ".format(torch.cuda.memory_cached(), torch.cuda.memory_allocated()))
    unet = unet.to(de)
    # initialize loss function and optimizer
    optimizer = Adam(params=unet.parameters(),lr = 0.001)
    # optimizer = SGD(params = unet.parameters(),lr = 0.01)

    # calculate steps per epoch for training and test set
    
    
    # trainSteps = len(train_dsl) // train_batch_size
    trainSteps = 10 #try changing this 
    
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.3,0.3,0.3])).to(de)
    dice_loss = DiceLoss()

    print("training the network...")
    # startTime = time.time()
    # H = {"Train_loss": [], "Train_Dice_coef": []}
    for epoch in tqdm(range(number_of_epochs)):
        unet.train(True)

        totalTrainLoss = 0
        totaDiceScore = 0
        # loop over the training set
        for batch_idx, (x, y) in tqdm(enumerate(train_dsl), leave=False):
            # send the input to the device
            torch.cuda.empty_cache()

            (x, y) = (torch.permute(x, [0,4,1,2,3]).to(de), torch.permute(y, [0,4,1,2,3]).to(de))
            
            # perform a forward pass and calculate the training loss
            optimizer.zero_grad()
            pred = unet(x.float())
            loss = criterion(pred,y)
            dc_score = dice_loss(pred, y)
                        
                        
            loss.backward()
            optimizer.step()

            totalTrainLoss += loss.item()    
            totaDiceScore += dc_score.item()           
            
            del x
            del y
            del loss
            del pred
            gc.collect()

        # calculate the average training
        avgTrainLoss = totalTrainLoss / len(train_dsl)
        avgDiceScore = totaDiceScore / len(train_dsl) 

        # update our training history
        # H["Train_loss"].append(avgTrainLoss)
        # H["Train_Dice_coef"].append(avgDiceScore)

       
        tot_val_loss = 0.0
        tot_val_score = 0.0
        with torch.no_grad():
            for i , (val_x, val_y) in enumerate(val_ds):
                (val_x, val_y) = (torch.permute(val_x, [0,4,1,2,3]).to(de), torch.permute(val_y, [0,4,1,2,3]).to(de))
                val_preds = unet(val_x)
                val_loss = criterion(val_y,val_preds)
                tot_val_loss += val_loss.item()
                val_dc_score = dice_loss(val_preds, val_y)
                tot_val_score += val_dc_score
        avg_val_loss = tot_val_loss // (i+1)
        avg_val_score = tot_val_score // (i+1)
        
        print("[INFO] EPOCH: {}/{} | Train loss: {:.6f},  Dice score : {:.4f} \n| Validation  loss: {:.6f},  Validation Dice score : {:.4f} ".format(epoch + 1, number_of_epochs,
            avgTrainLoss, avgDiceScore, avg_val_loss, avg_val_score), end="\n")
                    
    #saving the model
    try:
        torch.save(unet.state_dict(),"{}.pt".format(file_name))
    except:
        torch.save(unet.state_dict(), "test_only.pt")
        
    # endTime = time.time()
    
    # try:
    #     with open("{}.json".format(file_name), "w") as f:
    #         json.dump(H,f)
    # except TypeError:
    #     if(isinstance(avgTrainLoss), torch.Tensor):
    #         H['avgTrainLoss'] = H['avgTrainLoss'].cpu().detach().list()
            
            
        
    # print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime) , end="\n")
    return avgTrainLoss, avgDiceScore




def evaluate_model(model, test_loader, device, file_name):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        
        totalTestLoss = 0
        totaDiceScore = 0
        ce_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.3,0.3,0.3])).to(device)
        dice_loss = DiceLoss()
        # H = {"Test_loss": [], "Test_Dice_coef": []}

        for i, (x, y) in tqdm(enumerate(test_loader)):
            # send the input to the device
            (x, y) = (torch.permute(x, [0,4,1,2,3]).to(device), torch.permute(y, [0,4,1,2,3]).to(device))
            # make the predictions and calculate the validation loss
            pred = model(x.float())
            loss = ce_loss_fn(pred,y)
            dc_score = dice_loss(pred, y)
            
            
            totalTestLoss += loss.item()   
            totaDiceScore += dc_score 
            
            if ( i == test_batch_size):
                
                avgTestLoss = totalTestLoss // test_batch_size
                avgDiceScore = totaDiceScore //  test_batch_size
                # H["Test_loss"].append(avgTestLoss)
                # H["Test_Dice_coef"].append(avgDiceScore)
                del avgTestLoss
                del avgDiceScore
            
                if (i % 10 == 0 ):
                    print("[INFO] Test loss: {:.6f},  Dice score : {:.4f}".format(
                        avgTestLoss, avgDiceScore), end="\n") 
                    

            
            del x
            del y
            del loss
            torch.cuda.empty_cache()
            gc.collect()
        #------''' 
        
        # with open("{}.json".format(file_name), "w") as f:
        #     json.dump(H,f)

    return avgTestLoss, avgDiceScore, pred




def train_loop(dataloader, model, loss_fn, optimizer, device, file_name):
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    size = len(dataloader.dataset)
    dice_loss = DiceLoss()
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        (X, y) = (torch.permute(X, [0,4,1,2,3]).to(device), torch.permute(y, [0,4,1,2,3]).to(device))
        pred = model(X.float())
        loss = loss_fn(pred, y)
        dice_score = dice_loss(pred,y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            dice_score, loss, current = dice_score.item(),loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} dice score {dice_score:>4f} [{current:>5d}/{size:>5d}]")
    try:
        torch.save(model.state_dict(),"{}.pt".format(file_name))
    except:
        torch.save(model.state_dict(), "test_only.pt")
    print("cached data:{} memory allocated: {} ".format(torch.cuda.memory_cached(), torch.cuda.memory_allocated()))

    return dice_score, loss


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    dice_loss = DiceLoss()

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, dice_score = 0, 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            (X, y) = (torch.permute(X, [0,4,1,2,3]).to(device), torch.permute(y, [0,4,1,2,3]).to(device))

            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            dice_score += dice_loss(pred,y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 


def train_loop_dataset(dataset, model, loss_fn, optimizer, device, file_name):
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    size = len(dataset)
    dice_loss = DiceLoss()
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
        # Compute prediction and loss
    for i , (X,y) in tqdm(enumerate(dataset)):
        if i == 1:
            print(X.shape, y.shape)
        (X, y) = torch.permute(X, [3,0,1,2]).to(device), torch.permute(y, [3,0,1,2]).to(device)
        pred = model(X.float())
        loss = loss_fn(pred, y)
        dice_score = dice_loss(pred,y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        
        dice_score, loss = dice_score.item(),loss.item()
        print(f"loss: {loss:>7f} dice score {dice_score:>4f}")
    
    try:
        torch.save(model.state_dict(),"{}.pt".format(file_name))
    except:
        torch.save(model.state_dict(), "test_only.pt")
    print("cached data:{} memory allocated: {} ".format(torch.cuda.memory_cached(), torch.cuda.memory_allocated()))

    return dice_score, loss


def test_loop_dataset(dataset, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    dice_loss = DiceLoss()

    model.eval()
    size = len(dataset)
    test_loss, correct, dice_score = 0, 0, 0
    correct = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    for i , (X,y) in tqdm(enumerate(dataset)):

        (X, y) = torch.permute(X, [3,0,1,2]).to(device), torch.permute(y, [3,0,1,2]).to(device)
    
        pred = model(X.float())
        test_loss += loss_fn(pred, y).item()
        dice_score += dice_loss(pred,y).item()
    
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= i
    correct /= i
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 