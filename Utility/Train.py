

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:56:34 2022

@author: Family
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np 
from torch.utils.tensorboard import SummaryWriter

from .Dataset import AEDataset
from .AE import AE

def TrainAE(Weights,Weights_Test, Layer):
    
    "Set the model and its parameters"
    model_size=1
    for i in Weights[0][0][Layer].shape:
        model_size*=i
    criterion = nn.MSELoss()
    net=AE(input_shape=model_size)
    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=0.01)    
    schedule=[100*i for i in range(1,30)] #number of epochs to decrease the learning rate
    gamma= 0.6 #learning rate multiplier
    Batch_Size_Train=4
    Max_Epoch=100
    GPU =False #GPU
    
    # if the GPU is available load the model to GPU
    if GPU & torch.cuda.is_available():
        torch.cuda.empty_cache() #clear memory
        net = net.to('cuda')
    writer = SummaryWriter(comment=f"LR_{0.001}_WithoutOneAcc_BATCH_{Batch_Size_Train}_Layer_{Layer}_MaxEpoch_{Max_Epoch}")
    
    "Load data"
    dataset=AEDataset(Weights,Layer)
    dataset_Test= AEDataset(Weights_Test, Layer)
    trainloader = DataLoader(dataset, batch_size=Batch_Size_Train, shuffle=True)
    Valid_Loader = DataLoader(dataset_Test, batch_size=1, shuffle=True)
    
    "Training and Validation"
    min_valid_loss = np.inf
    max_acc=-np.inf
    for epoch in range(Max_Epoch):  # loop over the dataset multiple times
        if epoch in schedule: #to reduce the learning rate
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*gamma

        training_loss = 0.0
      
        net.train()
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs = data

            # Transfer Data to GPU if available
            if GPU & torch.cuda.is_available():
              inputs[0] = inputs[0].to('cuda')
              inputs[1] = inputs[1].to('cuda')
              inputs[2] = inputs[2].to('cuda')

            optimizer.zero_grad()
            outputs = net( inputs[0],inputs[1])

                       
            loss = criterion(outputs, inputs[0])
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()
            if i%5==0:
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, training_loss))
            
        writer.add_scalar(f'Training_loss_Layer_{Layer}', training_loss / len(trainloader), epoch)
        
        print(f' AE Training Accuracy \t {training_loss/ len(trainloader)}')
        
        "Validation"
        valid_loss = 0.0
        valid_acc=0
        net.eval()
        for inputs in Valid_Loader:
            with torch.no_grad():
                torch.cuda.empty_cache() #clear memory
                # Transfer Data to GPU if available
                if GPU & torch.cuda.is_available():
                  inputs[0] = inputs[0].to('cuda')
                  inputs[1] = inputs[1].to('cuda')
                  inputs[2] = inputs[2].to('cuda')
                  
                # Forward Pass
                target = net(inputs[0],inputs[1])
    
                # Find the Loss
                loss = criterion(target,inputs[0])

                # Calculate Loss
                valid_loss += loss.item()
                
        valid_loss/=len(Valid_Loader)
        print(f'Epoch {epoch+1} \t\t AE Validation Loss: {valid_loss } ')
        writer.add_scalar(f'Validation_Layer_{Layer}', valid_loss, epoch)
        writer.add_scalar(f'Best_accuracy_Layer_{Layer}', min_valid_loss, epoch)
        
        if (valid_acc/len(Valid_Loader)) > max_acc:
            max_acc=valid_acc/len(Valid_Loader)
         
        if min_valid_loss > valid_loss:
            print(f' AE Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(net.state_dict(), 'AE_model_'+str(Layer)+'.pt')
        writer.flush()
    print('Finished training for layer',int(Layer/2+1))
    print('Min_Loss=', min_valid_loss)
    

if __name__ == '__main__': 
    pass