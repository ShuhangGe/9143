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
    parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
    parser.add_argument('--epoch',type = int, default = 5)
    parser.add_argument('--num_works', type=int, default=4, help='number of cpu')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--test_batch', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='./dataset_FashionMNIST', help='path to save the data')
    parser.add_argument('--device', type=str, default='gpu', help='optimizer of the model')
    name_all = '2_1'
    logging.basicConfig(level=logging.INFO,  
                        filename=f'{name_all}.log',
                        filemode='a', 
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        )
    lr_min = 1e-9
    lr_max = 1e-2
    #load parmaters
    args = parser.parse_args()
    epochs = args.epoch
    device = args.device
    if device =='gpu' or device=='GPU':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=args.data_path, train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root=args.data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_min)
    
    scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max, step_size_up=2000, mode='triangular2', gamma=0.99)
    criterion = nn.CrossEntropyLoss()

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        train_accuracy.append(100. * correct_train / total_train)
        train_loss.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_accuracy.append(100. * correct_val / total_val)
        val_loss.append(running_val_loss / len(val_loader))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.savefig("2_2.png")
