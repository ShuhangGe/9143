import torch
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
from MiMiGoogLeNet import MiniGoogLeNet
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
import logging


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='class')
    parser.add_argument('--lr', type=float, default=1e-2,  help='learning rate')
    parser.add_argument('--epoch',type = int, default = 5)
    parser.add_argument('--num_works', type=int, default=4, help='number of cpu')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--test_batch', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='./dataset_FashionMNIST', help='path to save the data')
    parser.add_argument('--device', type=str, default='gpu', help='optimizer of the model')
    parser.add_argument('--name_all', type=str, default='9143', help='name of the experiment')
    


    #load parmaters
    args = parser.parse_args()
    epochs = args.epoch
    device = args.device
    LR = args.lr
    if device =='gpu' or device=='GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    name_all = args.name_all
    # logging.basicConfig(level=logging.INFO,  
    #                     filename=f'{name_all}.log',
    #                     filemode='a', 
    #                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #                     )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=args.data_path, train=True, download=True, transform=transform)
    model = MiniGoogLeNet()
    model = model.to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    # val_loss = []
    # train_accuracy = []
    # val_accuracy = []
    length = len(train_loader)
    # batch_sizes = [2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_count = 2
        batch_target = 2
        train_num = 0
        for i, (inputs, labels) in enumerate(train_loader):
            print(f'epoch: {epoch}, train: total length: {length}, index: {i}')
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if batch_count==batch_target or i == (length-1):
                train_num+=1
                loss.backward()
                optimizer.step()
                batch_target = batch_target*2
                # _, predicted = outputs.max(1)
                total_train += labels.size(0)
                # correct_train += predicted.eq(labels).sum().item()
                running_loss += loss.item()
            batch_count += inputs.shape[0]
        # train_accuracy.append(100. * correct_train / total_train)
        train_loss.append(running_loss / train_num)


    plt.figure(figsize=(12, 4))

    plt.plot(train_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


    plt.savefig(f"{name_all}.png")
