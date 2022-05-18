# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:56:08 2022

@author: Mahdi
"""

from torch.utils.data import Dataset
import torch
import numpy as np

from .Flatten import Flatten

class AEDataset(Dataset):

    def __init__(self, Weights, Layer):
        self.Weights = Weights
        self.round= len(Weights)
        self.num_user=len(Weights[0])
        self.Layer=Layer

    def __len__(self):
        return self.round*self.num_user

    def __getitem__(self, idx):
        
        "output and reference"
        
        out=(self.Weights[round(np.floor(idx / self.num_user))][(idx % self.num_user)][self.Layer])
        out2=(self.Weights[round(np.floor(idx / self.num_user))][0][self.Layer])
        out=torch.tensor(Flatten(out)[0])
        out2=torch.tensor(Flatten(out2)[0])
        
        return out, out2