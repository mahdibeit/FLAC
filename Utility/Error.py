# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:03:18 2022

@author: Mahdi
"""
from functools import reduce
import numpy as np

from .Flatten import Flatten

def AnalysisError(error,mean_tresh,var_tresh):
    """
    error---> List: [[(aprox, accuracte)_user1, ...]_round1, .... , [(aprox, accuracte)_user1, ...]_round10]
    
    output ---> List: [(mean error, var error)_user1, ...]
    
    """
    K=4
    if len (error)<=K:
        print(f"Size of the error is not bigger than {K}")
        
    flatten_error=[[np.subtract(Flatten(aprox)[0], Flatten(accurate)[0]) for aprox, accurate in iteration] for iteration in error]
    mean= [reduce(np.add, iteration)/len(error) for iteration in zip(*flatten_error)]
    flatten_error_powered=[[np.power(np.subtract(Flatten(aprox)[0], Flatten(accurate)[0]),2) for aprox, accurate in iteration] for iteration in error]
    variance= [reduce(np.add, iteration)/len(error) for iteration in zip(*flatten_error_powered)]
    
    for mean_user in mean:
        if  max(mean_user)>mean_tresh:
            print('Maximum_mean', max(mean_user),'\n', mean_user)
            mean_condition=False
            break
    else:
        mean_condition=True
        
    for var_user in variance:
        if  max(var_user)>var_tresh:
            print('Maximum_var',max(var_user),'\n','in', var_user)
            var_condition=False
            break
    else:
        var_condition=True
    
    avg_mean=0
    for mean_user in mean:
        avg_mean+=np.average(mean_user)/len(mean)
        
    avg_var=0
    for var_user in variance:
        avg_var+=np.average(var_user)/len(mean)
    
    return avg_mean, avg_var, mean_condition, var_condition


if __name__=="__main__":
    pass
    
    
    
    
    
    
    