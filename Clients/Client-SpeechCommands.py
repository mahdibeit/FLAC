# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:36:15 2022

@author: Mahdi
"""

from collections import OrderedDict
import warnings
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common.logger import log
from logging import INFO
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import random
import argparse
import os
import torchaudio

warnings.filterwarnings("ignore", category=Warning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")



# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze()


def train(net, trainloader, epochs,seed):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    new_sample_rate = 8000
    sample_rate=16000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    net.train()
    for _ in range(epochs):
        for audio, labels in trainloader:
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            audio = transform(audio)
            optimizer.zero_grad()
            loss = criterion(net(audio), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    new_sample_rate = 8000
    sample_rate=16000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for audio, labels in testloader:
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            audio = transform(audio)
            outputs = net(audio)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class DatasetSplit(Dataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        samples=[i for i in range(len(dataset))]
        length=int(len(dataset)/10)
        myrandom=random.Random(0) #Fixiing the seed
        myrandom.shuffle(samples)
        self.idxs = samples[length*seed:length*seed+length]
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        out= self.dataset[self.idxs[item]]
        return out
    
class DatasetNonIID(Dataset):
    def __init__(self, dataset, seed,labels, index_to_label,train_indicator):
        labels=np.asarray([index_to_label(targets) for _, _,targets,*_ in dataset])
        index=np.arange(len(dataset))
        idxs_labels = np.vstack((index, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()][0]
        self.dataset = dataset
        length=int(len(dataset)/20) #two 10 lists 
        testlen=50 #hald of size of test set
        seed=int(seed)
        self.testlen=testlen
        if train_indicator==True:
            first_half = list(idxs_labels[testlen+length*seed:length*seed+length])
            second_half=list(idxs_labels[len(dataset)-(length*seed+length):len(dataset)-length*seed-testlen])
            self.idxs=first_half+second_half
        else:
            first_half = list(idxs_labels[length*seed:length*seed+testlen])
            second_half=list(idxs_labels[len(dataset)-(length*seed+testlen):len(dataset)-length*seed])
            self.idxs=first_half+second_half

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        out = self.dataset[self.idxs[item]]
        return out


def load_data(seed, IID):
    """Load SpeachCommands (training and test set)."""
    
    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("Clients/", download=True)
    
            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
    
            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]
    
    
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    
    
    def label_to_index(word):
    # Return the position of the word in labels
        return torch.tensor(labels.index(word))


    def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
        return labels[index]
    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
    
    
    def collate_fn(batch):
    
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
    
        tensors, targets = [], []
    
        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [label_to_index(label)]
    
        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets
    
    if IID==True:
        trainloader = DataLoader(DatasetSplit(train_set,seed), batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        testloader = DataLoader(DatasetSplit(test_set,seed), batch_size=32,collate_fn=collate_fn,pin_memory=True)
        num_examples = {"trainset": len(train_set), "testset": len(test_set)}
        return trainloader, testloader, num_examples
    else:
        concatdataset=ConcatDataset([train_set,test_set])
        trainloader = DataLoader(DatasetNonIID(concatdataset,seed,labels, index_to_label, True), batch_size=32, shuffle=True)
        testloader = DataLoader(DatasetNonIID(concatdataset,seed,labels, index_to_label,False), batch_size=32)
        num_examples = {"trainset": len(train_set), "testset": len(test_set)}
        return trainloader, testloader, num_examples




# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    
    iid=True
    
    """Fixing the seed for reproducability"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="set the seed for reporiducabilty")
    args = parser.parse_args()
    if args.seed:
        seed = args.seed
        log(INFO, f"Using seed {seed} for reproducability")
        torch.manual_seed(seed)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    
    """Create model, load data, define Flower client, start Flower client."""

    # Load data (Speech Command)
    trainloader, testloader, num_examples = load_data(int(args.seed),IID=iid)
    
    # Load model
    net = Net(n_input=1, n_output=35).to(DEVICE)
    
    if iid:
        log(INFO, "Using IID dataset")
    else:
        log(INFO, "Using Non-IID dataset")
    
    # Flower client
    class SpeechClient(fl.client.NumPyClient):
        def __init__(self,args):
            super().__init__()
            self.Global_round=0
            self.args=args
            
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader,  1, self.args.seed,)

            """Utilize the seed number as the IID number
            [Seed Number, Parameters]
            """
            return self.get_parameters(), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("localhost:8080", client=SpeechClient(args))


if __name__ == "__main__":
    main()