# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:40:03 2023

@author: aysha
"""
from UNet3D import UNet3D
import torch


from tqdm import tqdm
from HelperFunctions import evaluate_model, print_size_of_model
from DataLoader import test_mri_dataset
from torch.utils.data import Subset, DataLoader
from torch.quantization.observer import MinMaxObserver,MovingAverageMinMaxObserver,PerChannelMinMaxObserver
import torchmetrics

def evaluate(model, data_loader):
    model.eval()
    dice_scores = list()
    losses = list()
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image = torch.permute(image, [0,4,1,2,3])
            target = torch.permute(target, [0,4,1,2,3])
            output = model(image)
            s_output = torch.nn.Softmax(dim=1)(output)#(1,4,128,128,128)
           
            target_indices = torch.argmax(target, dim=1) #(1,128,128,128)
            
            output_indices =  torch.argmax(s_output, dim=1)
            dc_score  = torchmetrics.functional.dice(output_indices,target_indices,
                                                           ignore_index=0,
                                                          num_classes=4, average='macro')
            diceloss = 1 - dc_score
            weight = torch.Tensor([0.00414074, 0.53114366, 0.12849232, 0.33622329])
            ce_loss = torch.nn.functional.cross_entropy(output, target_indices, weight=weight)
            loss = ce_loss + 0.01*diceloss            
            dice_scores.append(dc_score.item())
            losses.append(loss.item())
    return losses, dice_scores   
import os
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')      

if __name__ == '__main__':
    # Quantization Aware Training
    print("backends:",torch.backends.quantized.supported_engines)

    qat_model = UNet3D(3, 4, quantize=True)
    file_name = "results_05112023/2023-11-05-13-29_checkpoint.pt"

    qat_model.load_state_dict(torch.load(file_name))
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
    evaluate(qat_model,DataLoader(Subset(test_mri_dataset,range(15,18)), batch_size=2))
    print("inference over")
    
    qat_model.cpu()
    qat_model = torch.quantization.convert(qat_model,inplace=True)
    file_name = "results_05112023/2023-11-05-13-29_checkpoint_qat_model.pt"
    torch.save(qat_model.state_dict(),file_name)
    losses, dice_score = evaluate(qat_model,DataLoader(Subset(test_mri_dataset,range(25,60)), batch_size=2))    
    print("loss:", losses,"dice: ", dice_score)
    print_size_of_model(qat_model)
    
    
    model = UNet3D(3, 4)
    model.load_state_dict(torch.load('results_05112023/2023-11-05-13-29_checkpoint.pt'))
    losses_un, dice_score_un = evaluate(model,DataLoader(Subset(test_mri_dataset,range(25,60)), batch_size=2))    
    print("loss_un:", losses_un,"dice_un: ", dice_score_un)
    print_size_of_model(model)
        
   