# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 00:55:32 2022

@author: Mahdi
"""



import torch

import torch.nn as nn
from .Flatten import Flatten, DeFlatten
from .AE import AE




def AE_Inference(inputs, layer ,ref):
    with torch.no_grad():
        "Set the model"
        model_size=1
        for i in inputs.shape:
            model_size*=i
        net=AE(input_shape=model_size)
        net = net.double()
        criterion = nn.MSELoss()
        net.load_state_dict(torch.load('AE_model_'+str(layer)+'.pt'))

         
        "Validation"
        net.eval()     # Optional when not using Model Specific layer
        inputs, shape= Flatten(inputs)
        ref, _ =Flatten(ref)

        # Forward Pass
        inputs=torch.tensor(inputs) 
        ref=torch.tensor(ref) 
        target = net(inputs[None,:],ref[None,:]) #changing the shape to incoperate the batch
        target=torch.squeeze(target) #Reshaping it back
    
        #print(pred, labels.tolist())
        # print(criterion(inputs,target).item())
        target=DeFlatten(target.detach().numpy(), shape)
    return target