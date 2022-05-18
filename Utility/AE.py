# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:59:45 2022

@author: Mahdi
"""


import torch.nn as nn
import torch.nn.functional as F



class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        '''Encoder and decoder part of the autoencoder'''
        self.size=kwargs["input_shape"]
        self.encoder1 = nn.Linear(in_features=kwargs["input_shape"], out_features=int(kwargs["input_shape"]/32))

        self.decoder2 = nn.Linear(in_features=int(kwargs["input_shape"]/32), out_features=int(kwargs["input_shape"]))
        # self.decoder2 = nn.Linear(in_features=int(kwargs["input_shape"]/16)+int(kwargs["input_shape"]), out_features=int(kwargs["input_shape"]))
        
    def forward(self, x,y):
        x = F.relu(self.encoder1(x))
        
        # x = (self.decoder2(torch.cat((x,y),dim=1)))
        x = (self.decoder2(x))     
        return x
        