# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:23:53 2022

@author: Mahdi
"""
import numpy as np

def Flatten(w_w):
    flatten_weighted_weights=np.array([])
    map_shape={"Seperation":[0], "Shape":[]}  
    tot=0
    for  layers in w_w:
        temp=1
        flatten_weighted_weights=np.append(flatten_weighted_weights, layers.flatten())
        # print(layers.shape)
        map_shape["Shape"].append(layers.shape)
        for i in layers.shape:
            temp=temp*i
        tot+=temp
        map_shape["Seperation"].append(tot)
    return flatten_weighted_weights, map_shape

def DeFlatten(f_w_w, map_shape):
    out=[]
    for idx, shape in enumerate(map_shape["Shape"]):
        out.append(f_w_w[map_shape["Seperation"][idx]:map_shape["Seperation"][idx+1]].reshape(shape))
    for  layers in out:
        # print("outputcheck",layers.shape)
        pass
    return out