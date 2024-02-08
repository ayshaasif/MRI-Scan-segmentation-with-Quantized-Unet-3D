# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:51:25 2023

@author: aysha
"""
#--------------------Helper Functions --------------------------------

from UNet3D import UNet3D
import torch
import matplotlib.pyplot as plt
import numpy as np
# from metrics import get_sens_spec_df
import torch.nn.functional as F
from DataLoader import test_mri_dataset,test_data_loader,sample_ds
import torchmetrics
from torchsummary import summary
from tqdm import tqdm 

def evaluate(model, data_loader):
    precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=4)
    model.eval()
    dice_scores = list()
    losses = list()
    with torch.no_grad():
        for images_batch, targets_batch in tqdm(data_loader):
            images = torch.permute(images_batch, [0,4,1,2,3])
            targets = torch.permute(targets_batch, [0,4,1,2,3])
            output = model(images)
            s_output = torch.nn.Softmax(dim=1)(output)#(1,4,128,128,128)
           
            target_indices = torch.argmax(targets, dim=1) #(1,128,128,128)
            
            output_indices =  torch.argmax(s_output, dim=1)
            dc_score  = torchmetrics.functional.dice(output_indices,target_indices,
                                                           # ignore_index=0,
                                                          num_classes=4, average='macro')
            diceloss = 1 - dc_score
            weight = torch.Tensor([0.00414074, 0.53114366, 0.12849232, 0.33622329])
            ce_loss = torch.nn.functional.cross_entropy(output, target_indices, weight=weight)
            loss = ce_loss + 0.01*diceloss            
            dice_scores.append(dc_score.item())
            losses.append(loss.item())
            precision_score = precision(output_indices, target_indices)
            print(f"precision: {precision_score}")
    return losses, dice_scores  


if __name__ == '__main__':    

    #----------------------------------------Testing--------------------------------------------------------------------
   
    h = len(test_mri_dataset)
    row = np.random.randint(h)

    print("image: ", str(row))
    img = sample_ds[row][0] #in case of dataset first 0 -> 0 row, second 0 is 0 column
    mask = sample_ds[row][1] # 0 row, 1 column
    print(img.shape, mask.shape)
    file_name = "results_19122023/2023-12-19-23-44_checkpoint"

    unet = UNet3D(in_channels=3, num_classes=4,quantize=True)
    unet.load_state_dict(torch.load("{}.pt".format(file_name)),strict=False)
    # summary(unet, (3,128,128,128), batch_size=4,device="cpu")
    print("params loaded")
    cpu_device = torch.device("cpu:0")
    unet = unet.float()
    unet.eval()
    losses, dice_scores =  evaluate(unet, test_data_loader)
    
    # '''
    
    img_sqz = torch.unsqueeze(img,dim=0) #because the model takes in batch size as well
    print("img_sqz shape: ",img_sqz.shape) #torch.Size([1, 128, 128, 128, 3])
    
    pred = unet(torch.permute(img_sqz,[0,4,1,2,3])) 
    print("prediciton 1 : ",pred.shape) # torch.Size([1, 4, 128, 128, 128])
    pred = torch.nn.Softmax(dim=1)(pred)
    argmax_mask = torch.argmax(mask, dim=3)
  
    pred = torch.permute(pred,[0,2,3,4,1]) # re-ordering the channel
    print("permuted_pred shape: ", pred.shape) #torch.Size([1, 128, 128, 128, 4])


    sqz_pred = torch.squeeze(pred).detach() # numpy array [128,128,128,4]
    
    
    
    argmax_pred = torch.argmax(sqz_pred,axis=3) 
    print(mask.shape, argmax_pred.shape)
    dc_score  = torchmetrics.functional.dice(argmax_pred,argmax_mask.int(),
                                              average='none',
                                              # ignore_index=0,
                                              num_classes=4
                                              )
    print(f"dice score: {dc_score}")
    
    precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=4)
    precision_score = precision(argmax_pred, argmax_mask.int())
    
    print(f"precision: {precision_score}")

    # -----------------plotting the predictions-------------------------------
    fig,ax = plt.subplots(1,3)
    print(file_name)
        
    # plt.subplot(1,3,1)
    ax[0].set_title("MRI scan")
    ax[0].imshow(img[100,:,:,1], cmap="gray")
    
    # plt.subplot(1,3,2)
    ax[1].set_title("ground truth")
    mask = torch.argmax(mask, axis=3)
    ax[1].imshow(mask[100,:,:])
    
    # plt.subplot(1,3,3)
    ax[2].set_title("predicted mask")
    ax[2].imshow(argmax_pred[100,:,:])
    for x in ax:x.axis("off")
    
    
    
    fig,ax = plt.subplots(1,3) 
    # plt.subplot(1,3,1)
    ax[0].set_title("MRI scan")
    ax[0].imshow(img[:,:,90,1], cmap="gray")
    
    # plt.subplot(1,3,2)
    ax[1].set_title("ground truth")
    ax[1].imshow(mask[:,:,90])
    
    # plt.subplot(1,3,3)
    ax[2].set_title("predicted mask")
    ax[2].imshow(argmax_pred[:,:,90])
    for x in ax:x.axis("off")


    del(img_sqz)
    del(pred)
    del(sqz_pred)
    del(mask)
    del(img)
    del(test_mri_dataset)
    
'''
#     ----------------- Evaluate the model on test data ------------------------
#     loss,accuracy,pred = tqdm(evaluate_model(model=unet, test_loader=test_data_loader, device=torch.device("cpu:0"), file_name=file_name))
#     print("Test Dice Score : {:.4f} , Test loss : {:.4f}".format(accuracy, loss))
# '''