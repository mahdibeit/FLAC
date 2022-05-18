# -*- coding: utf-8 -*-
"""
Created on Tue May 17 21:07:44 2022

@author: Mahdi
"""

'''
------------------------------------------------------------------------------
Copyright 2022 Mahdi Beitollahi
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------
'''

from typing import  Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.client_proxy import ClientProxy
from functools import reduce
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
import copy

from Utility import TrainAE
from Utility import AE_Inference
from Utility import AnalysisError


class History():
    def __init__(self):
        self.list= []  
        self.round=0
    def update(self, weighted_weights):
        copied_list=copy.deepcopy(weighted_weights)
        self.list.append(copied_list)
        self.round +=1
        
        print('\n', "Gobal Round", self.round)
    def Len(self):
        print(len(self.list))
    
    each_round_weighted_weights=[]
   
    

def aggregateNew(results: List[Tuple[Weights, int]],client_id, flac_parameters) -> Weights:
    """Compute weighted average."""

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * 1 for layer in weights] for weights, num_examples in results
    ]
    
    #sort and match the id with weights
    weighted_weights = [weights for _,weights in sorted(zip(client_id,weighted_weights),key=lambda pair: pair[0])]


    """ Saved File:
        First Index: Round 
        Second Index: User
        Third Index: Layer
    """
    flac_parameters['history'].update(weighted_weights)
    with open("history", "wb") as fp:
        pickle.dump(flac_parameters['history'].list, fp)
    

    q=flac_parameters['q']
    Layers=flac_parameters['layers']
    history=flac_parameters['history']
    number_of_users=flac_parameters['number_of_users']
    writer=flac_parameters['Writer']
    
    """Training State"""
    if flac_parameters['State']=="Training":
        writer.add_scalar('Compression', 0, flac_parameters['history'].round)
        if flac_parameters['wait_q_rounds']<q:
            print('Sending without Compression')
            flac_parameters['wait_q_rounds'] +=1
        else:
            print('start training')
            flac_parameters['history'].list_acc=copy.deepcopy(flac_parameters['history'].list)
            for layer in Layers:
                TrainAE(history.list_acc[history.round-q:history.round-2], [history.list_acc[history.round-1]],layer)
            flac_parameters['State']="Compression"
            flac_parameters['first_time_flag']=True
    
    
    
    
    """Compression State"""
    weighted_weights_accu=copy.deepcopy(weighted_weights)
    weighted_weights_train=copy.deepcopy(weighted_weights)
    weighted_weights_res=copy.deepcopy(weighted_weights)
    
    mean_tresh=0.5
    var_tresh=0.008
    if flac_parameters['State']=="Compression":
        print('Sending with Compression')
        writer.add_scalar('Compression', 1, flac_parameters['history'].round)
        if flac_parameters['first_time_flag']==True:
            flac_parameters['error']=[]
            flac_parameters['first_time_flag']=False
            
        sub_error=[] 
        for idx, w in enumerate(weighted_weights_train[:]): # Do not forget to change this
            for layer in Layers:
                
                weighted_weights_res[idx][layer]= (AE_Inference(w[layer],layer,weighted_weights_accu[0][layer]))
                # weighted_weights_res[idx][layer]= (AE_Inference(w[layer],layer,history.list[4][idx][layer]))
                # print('round',history.round,'Idx',idx,'layer',layer)
                # print('accur',weighted_weights_accu[0][layer][0][:5])
                # print('before', weighted_weights[idx][layer][0][:5])
                # print('after', weighted_weights_res[idx][layer][0][:5])
                sub_error.append((weighted_weights_res[idx][layer], weighted_weights_accu[idx][layer]))
                          
        flac_parameters['error'].append(sub_error)
        if len(flac_parameters['error'])>=q:
            mean, var, mean_condition, var_condition= AnalysisError(flac_parameters['error'][len(flac_parameters['error'])-q:],mean_tresh,var_tresh)
            print('--------------------')
            print('AVG_mean', mean, 'AVG_var',var)
            print('mean_condition', mean_condition, 'var_condition',var_condition)
            writer.add_scalar('AVG_mean', mean, flac_parameters['history'].round)
            writer.add_scalar('AVG_var', var, flac_parameters['history'].round)
            print('--------------------')
            
            if mean_condition & var_condition == False:
                flac_parameters['State']="Training"
                flac_parameters['wait_q_rounds']=0
                    
        weighted_weights=copy.deepcopy(weighted_weights_res)
        
        
    """Aggregate"""
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / number_of_users #num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime

class FLAC(FedAvg):
    def __init__(self,  *args, q=5, layers=[1], **kwargs):
        super(FLAC, self).__init__(*args, **kwargs)
        self.writer = SummaryWriter(comment= "FLAC") 
        self.flac_parameters=dict([("error",[]), ("history", History()), ("first_time_flag", False),
                                   ('q',q), ('layers', layers), ('State',"Training"), ("Writer", self.writer),
                                   ('wait_q_rounds', 0), ('number_of_users', kwargs["min_fit_clients"])] )
    
    """Save and graph the aggregated loss and accuraies"""
    def aggregate_evaluate(
    self,
    rnd: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        loss = [r.loss * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        loss_aggregated = sum(loss) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
        self.writer.add_scalar('accuracy_aggregated', accuracy_aggregated, rnd)
        self.writer.add_scalar('loss_aggregated', loss_aggregated, rnd)
        
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
            
    def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using weighted average."""
            client_id=[int(client.cid[11:]) for client, _ in results]
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}
            # Convert results
            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            return weights_to_parameters(aggregateNew(weights_results,client_id, self.flac_parameters)), {}
