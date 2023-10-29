import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable,Function
from lenet import LeNet5_BatchNorm_2,LeNet5_BatchNorm_3,LeNet5_Dropout_bn,LeNet5_Dropout
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_batchnorm_parameters(model,name_all):
    weights = []
    biases = []
    layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            layer_names.append(name)
            weights.append(module.weight.data.cpu().numpy())
            biases.append(module.bias.data.cpu().numpy())

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    sns.violinplot(data=weights, ax=ax[0], palette='muted', inner="quartile")
    ax[0].set_title('BatchNorm Weights Distribution')
    ax[0].set_xticklabels(layer_names)

    sns.violinplot(data=biases, ax=ax[1], palette='muted', inner="quartile")
    ax[1].set_title('BatchNorm Biases Distribution')
    ax[1].set_xticklabels(layer_names)

    plt.tight_layout()
    plt.savefig(f"{name_all}.png") 


if __name__ == '__main__':

    print('start')
    parser = argparse.ArgumentParser(description='class')
    parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
    parser.add_argument('--epoch',type = int, default = 5)
    parser.add_argument('--num_works', type=int, default=4, help='number of cpu')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./dataset3', help='path to save the data')
    parser.add_argument('--device', type=str, default='gpu', help='optimizer of the model')
    parser.add_argument('--name_all', type=str, default='gpu', help='optimizer of the model')


    #load parmaters
    args = parser.parse_args()
    name_all = args.name_all
    logging.basicConfig(level=logging.INFO,  
                        filename=f'{name_all}.log',
                        filemode='a', 
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        )
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
    logging.info('paramaters set')
    #use my dataloader
    
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32,padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) ,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) ,
        ])
    train_dataset =  torchvision.datasets.MNIST(root=args.data_path,train=True,download=True,transform = transform_train)
    test_dataset =  torchvision.datasets.MNIST(root=args.data_path,train=False,download=True,transform = transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=num_works)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=num_works)

    logging.info('data ready')
    #model
    model = LeNet5_BatchNorm_3()

    weight_decay = 5e-4
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss()

    model = model.to(device)
    logging.info(model)
    logging.info('start train')
    length_train = len(train_loader)
    length_test = len(test_loader)
    #train
    correct_total = 0
    best_acc = 0
    loss_total = 0
    best_acc = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fun(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(100. * correct / total)
        # Testing
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = loss_fun(outputs, target)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        test_losses.append(test_loss/len(test_loader))
        test_accuracies.append(100. * correct / total)

    logging.info(f"Final Training Loss: {train_losses[-1]}")
    logging.info(f"Final Test Loss: {test_losses[-1]}")
    logging.info(f"Final Train Accuracy: {train_accuracies[-1]}")
    logging.info(f"Final Test Accuracy: {test_accuracies[-1]}")

    # Output BatchNorm Parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            logging.info(f"\nLayer: {name}")
            logging.info(f"parameters: {module.state_dict()}")
            logging.info(f"Mean: {module.running_mean}")
            logging.info(f"Variance: {module.running_var}")
            logging.info(f"Gamma (Weight): {module.weight.data}")
            logging.info(f"Beta (Bias): {module.bias.data}")
    plot_batchnorm_parameters(model,name_all)







