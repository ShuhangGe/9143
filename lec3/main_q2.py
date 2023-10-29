import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable,Function
import time
from MiMiGoogLeNet import MiniGoogLeNet
import torchvision
import matplotlib.pyplot as plt
import logging


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='class')
    parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
    parser.add_argument('--epoch',type = int, default = 5)
    parser.add_argument('--num_works', type=int, default=4, help='number of cpu')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='./dataset_FashionMNIST', help='path to save the data')
    parser.add_argument('--device', type=str, default='gpu', help='optimizer of the model')
    name_all = '2_1'
    logging.basicConfig(level=logging.INFO,  
                        filename=f'{name_all}.log',
                        filemode='a', 
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        )

    #load parmaters
    args = parser.parse_args()
    epochs = args.epoch
    device = args.device
    if device =='gpu' or device=='GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    train_batch = args.train_batch
    test_batch = args.test_batch
    num_works = args.num_works
    LR =args.lr
    # logging.info('paramaters set')
    model = MiniGoogLeNet()
    loss_fun = nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    candidate_lrs = np.logspace(-9, 1, 10)  # 10 candidate learning rates between 10^-9 and 10^1
    losses = []
    length = len(train_loader)
    for lr in candidate_lrs:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        total_loss = 0
        for epoch in range(5):  # Train for 5 epochs
            for i, (images, labels) in enumerate(train_loader):
                print(f'epoch: {epoch}, train: total length: {length}, index: {i}')
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        losses.append(total_loss / (5 * len(train_loader)))  # Average loss over 5 epochs

    # Plotting
    plt.plot(candidate_lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Loss')
    plt.savefig("lr.png") 



