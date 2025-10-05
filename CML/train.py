
 

from UNet import UNet3D
from DiceLoss import DiceLoss
from FocalLoss import FocalLoss
from IOULoss import IoULoss
from FocalTverskyLoss import FocalTverskyLoss
from BRaTsDataset import BraTSDataset

import torch
import torch.nn as nn
from torch.optim import Adam

import os
import numpy as np
import pandas as pd
import glob as glob
import time
from  torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

import gc
# Release unused GPU cache memory
torch.cuda.empty_cache()

# Invoke Python's garbage collector to reclaim unreachable memory
gc.collect()

BASE_DIR = "BratsDataset"
IMG_DIR = "images"
MASK_DIR = "masks"
train_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,transform = torch.from_numpy)
 
test_mri_dataset = BraTSDataset(root_dir = BASE_DIR, image_dir = IMG_DIR, mask_dir = MASK_DIR,test=True,transform = torch.from_numpy)


# sub set of the dataset for quick testing
indices = list(range(5))
train_mri_dataset = torch.utils.data.Subset(train_mri_dataset, indices)
test_mri_dataset = torch.utils.data.Subset(test_mri_dataset, indices)
np.random.seed(42)  
print("shape of train_mri_dataset[0][0]: ", train_mri_dataset[0][0].shape  ," shape of train_mri_dataset[0][1]: ", train_mri_dataset[0][1].shape)
print("Number of training samples:", len(train_mri_dataset))
print("Number of testing samples:", len(test_mri_dataset))  


pin_memory = False

train_data_loader = DataLoader(train_mri_dataset, batch_size=2,
    num_workers=2, pin_memory=pin_memory,shuffle=pin_memory)


test_data_loader = DataLoader(test_mri_dataset, batch_size=2,
    num_workers=2, pin_memory=pin_memory) 



SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze()
    intersection = torch.mul(outputs,labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs + labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded

def evaluate_model(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        
        totalTestLoss = 0
        totalIOUscore = 0
        # iou_loss_fn = IoULoss()
        ce_loss_fn = nn.CrossEntropyLoss()
        dice_loss_fn = DiceLoss()
        # fc_loss_fn = FocalLoss()
        fct_loss_fn = FocalTverskyLoss()
        print("loss functions used: {}, {}, {}".format( ce_loss_fn,fct_loss_fn, dice_loss_fn))
        for (x, y) in tqdm(test_loader):
            # send the input to the device
            (x, y) = (torch.permute(x, [0,4,1,2,3]).to(device), torch.permute(y, [0,4,1,2,3]).to(device))
            # make the predictions and calculate the validation loss
            pred = model(x.float())
            ce_loss = ce_loss_fn(pred,y)
            # iou_loss = iou_loss_fn(pred,y)
            dice_loss = dice_loss_fn(pred, y)
            # fc_loss = fc_loss_fn(pred,y)
            fct_loss = fct_loss_fn(pred,y)
            loss =   ce_loss + fct_loss + dice_loss
            totalTestLoss += loss    
            iou_score = iou_pytorch(pred,y)
            totalIOUscore += iou_score
    
        avgTestLoss = totalTestLoss / (len(test_loader))
        avgIOUscore = totalIOUscore / (len(test_loader))
   

    return avgTestLoss, avgIOUscore


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')



# ... (assuming DiceLoss, FocalTverskyLoss, iou_pytorch, and UNet3D are defined)

def train(unet, train_dsl, number_of_epochs, file_name, de="cuda", accumulation_steps=8):
    # Set CUDA_VISIBLE_DEVICES to use the NVIDIA GPU (device ID 1 on your system)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Move model to device once, outside the loop
    unet.to(de)
    
    # Initialize optimizer, scaler, and loss functions
    learning_rate = 0.001
    optimizer = Adam(params=unet.parameters(), lr=learning_rate)
    scaler = GradScaler()

    dice_loss_fn = DiceLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    fct_loss_fn = FocalTverskyLoss()

    print("Training the network...")
    startTime = time.time()

    # Get the total number of batches
    total_batches = len(train_dsl)
    
    for e in tqdm(range(number_of_epochs)):
        unet.train()
        total_train_loss = 0.0
        total_iou_score = 0.0
        
        # Zero gradients at the beginning of the epoch
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(train_dsl):
            # Move input and labels to device
            # Permute dimensions if necessary (adjust based on your data loader)
            x = torch.permute(x, [0, 4, 1, 2, 3]).to(de)
            y = torch.permute(y, [0, 4, 1, 2, 3]).to(de)

            # --- AMP and Gradient Accumulation Logic ---
            with autocast(device_type=de):
                # Forward pass
                pred = unet(x.float())
                
                # Calculate combined loss
                dice_loss = dice_loss_fn(pred, y)
                ce_loss = ce_loss_fn(pred, y)
                fct_loss = fct_loss_fn(pred, y)
                loss = ce_loss + dice_loss
                
                # Normalize the loss for gradient accumulation
                loss = loss / accumulation_steps

            # Backward pass (with scaler)
            scaler.scale(loss).backward()
            
            # --- Update weights and metrics ---
            iou_score = iou_pytorch(y, pred)
            total_iou_score += iou_score.mean().item()
            total_train_loss += loss.item() * accumulation_steps  # Un-normalize loss for display
            
            # Check if it's time to update weights
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i == 0 and e == 0:
                print("Pred shape: ", pred.shape)
                print("X shape: ", x.shape)
                print("Y shape: ", y.shape)

        # Handle any remaining mini-batches at the end of the epoch
        if (i + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Calculate average loss and IOU score
        avg_train_loss = total_train_loss / total_batches
        avg_iou_score = total_iou_score / total_batches

        print(f"[INFO] EPOCH: {e + 1}/{number_of_epochs}")
        print(f"Train loss: {avg_train_loss:.6f}, IOU score: {avg_iou_score:.4f}")

    # Save the model
    try:
        torch.save(unet.state_dict(), file_name)
    except Exception as e:
        print(f"Error saving model: {e}")
        torch.save(unet.state_dict(), "test_only.pt")
        
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    # print_size_of_model(unet) # Assuming this is a custom function



if __name__ == "__main__":
    from datetime import datetime
    unet = UNet3D(in_channels=3, num_classes=4, quantize=False)
    print(unet)
    # sample_input = torch.randn(1, 3, 128, 128, 128)

    # Now, test the model with the correctly shaped input
    # try:
    #     test_unet = unet(sample_input)
    #     print("Model forward pass was successful!")
    #     print("Output shape:", test_unet.shape)
    # except AttributeError as e:
    #     print("An AttributeError occurred:", e)


#   Running the actual training
    print(train_mri_dataset[0][0].shape, train_mri_dataset[0][1].shape)

    train(unet,train_data_loader,number_of_epochs=1,file_name=f'unet3d_{datetime.now().strftime("%Y-%m-%d %H_%M")}.pt',de="cuda")

