# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:20:31 2023

@author: aysha
"""


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
from torch.utils.data import DataLoader, Subset

from torch.quantization.observer import MinMaxObserver,MovingAverageMinMaxObserver,PerChannelMinMaxObserver

if __name__ == '__main__':    
    print("backends:",torch.backends.quantized.supported_engines)
    
    qat_model = UNet3D(3, 4, quantize=True)
    qat_model.to("cpu")
    qat_model.eval()
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig("onednn") #backend
    qat_model.qconfig = torch.quantization.QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8,qscheme=torch.per_tensor_affine),
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
    print("qat_model:", qat_model)
    
    qat_model.train()
    qat_model= torch.quantization.prepare_qat(qat_model, inplace=True)
    
    print("inference started ............")
    with torch.inference_mode():
        for image, _ in tqdm(DataLoader(Subset(test_mri_dataset,range(15,18)), batch_size=2)):
            image = image.permute([0,4,1,2,3])
            qat_model(image)
    print("inference over")
    
    qat_model.cpu()
    qat_model = torch.quantization.convert(qat_model,inplace=True)
    file_name = "results_05112023/2023-11-05-13-29_checkpoint_qat_model.pt"
    qat_model.load_state_dict(torch.load(file_name))
    
    #----------------------------------------Testing--------------------------------------------------------------------
    from DataLoader import test_mri_dataset

    h = len(test_mri_dataset)
    row = np.random.randint(h)

    print("image: ", str(row))
    img = sample_ds[row][0] #in case of dataset first 0 -> 0 row, second 0 is 0 column
    mask = sample_ds[row][1] # 0 row, 1 column
    print(img.shape, mask.shape)

    
    print("params loaded")
    cpu_device = torch.device("cpu:0")

    
    # '''
    
    img_sqz = torch.unsqueeze(img,dim=0) #because the model takes in batch size as well
    print("img_sqz shape: ",img_sqz.shape) #torch.Size([1, 128, 128, 128, 3])
    
    pred = qat_model(torch.permute(img_sqz,[0,4,1,2,3])) 
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
                                              ignore_index=0,
                                              num_classes=4
                                              )
    print(f"dice score: {dc_score}")
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
    
