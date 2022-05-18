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
from torchvision.datasets import CIFAR10
import numpy as np
import random
import argparse

warnings.filterwarnings("ignore", category=Warning)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs,seed):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
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
        length=int(len(dataset)/10)
        samples=[i for i in range(len(dataset))]
        myrandom=random.Random(0) #Fixiing the seed
        myrandom.shuffle(samples)
        self.idxs = samples[length*seed:length*seed+length]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class DatasetNonIID(Dataset):
    def __init__(self, dataset, seed,train_indicator):
        labels=np.asarray([targets for _ ,targets in dataset])
        index=np.arange(len(dataset))
        idxs_labels = np.vstack((index, labels))
        idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()][0]
        self.dataset = dataset
        length=int(len(dataset)/20) #two 10 lists 
        testlen=500 #hald of size of test set
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

        image, label = self.dataset[self.idxs[item]]
        return image, label


def load_data(seed, IID):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("Clients/dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("Clients/dataset", train=False, download=True, transform=transform)
    if IID==True:
        trainloader = DataLoader(DatasetSplit(trainset,seed), batch_size=32, shuffle=True)
        testloader = DataLoader(DatasetSplit(testset,seed), batch_size=32)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}
        return trainloader, testloader, num_examples
    else:
        concatdataset=ConcatDataset([trainset,testset])
        trainloader = DataLoader(DatasetNonIID(concatdataset,seed,True), batch_size=32, shuffle=True)
        testloader = DataLoader(DatasetNonIID(concatdataset,seed,False), batch_size=32)
        num_examples = {"trainset": len(trainset), "testset": len(testset)}
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

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    trainloader, testloader, num_examples = load_data(int(args.seed),IID=iid)
    if iid:
        log(INFO, "Using IID dataset")
    else:
        log(INFO, "Using Non-IID dataset")
    
    # Flower client
    class CifarClient(fl.client.NumPyClient):
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
    fl.client.start_numpy_client("localhost:8080", client=CifarClient(args))


if __name__ == "__main__":
    main()