import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable,Function
import time
from resnet import ResNet18
import torchvision
from torchvision import transforms


if __name__ == '__main__':
    print('start')
    parser = argparse.ArgumentParser(description='class')
    parser.add_argument('--lr', type=float, default=0.1,  help='learning rate')
    parser.add_argument('--epoch',type = int, default = 5)
    parser.add_argument('--num_works', type=int, default=2, help='number of cpu')
    parser.add_argument('--train_batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--data_path', type=str, default='./dataset3', help='path to save the data')
    parser.add_argument('--device', type=str, default='gpu', help='optimizer of the model')
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
    print('paramaters set')
    #use my dataloader
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    train_dataset =  torchvision.datasets.CIFAR10(root=args.data_path,train=True,download=True,transform = transform_train)
    test_dataset =  torchvision.datasets.CIFAR10(root=args.data_path,train=False,download=True,transform = transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch,
                                               shuffle=True,
                                               num_workers=num_works)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch,
                                               shuffle=False,
                                               num_workers=num_works)

    print('data ready')
    model = ResNet18()

    weight_decay = 5e-4
    momentum = 0.9
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss()

    model = model.to(device)
    #print(model)
    print('start train')
    length_train = len(train_loader)
    length_test = len(test_loader)
    epoch_time = 0
    train_num = 0
    for epoch in range(epochs):
        print(f'epoch: {epoch}----------------------------------------------------------------')
        #train
        for index, data in enumerate(train_loader):
            print(f'train: total length: {length_train}, index: {index}')
            model.train()
            img, label = data
            start = time.time()
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fun(output,label)
            print('loss_train: ',loss.item())
            loss.backward()
            optimizer.step()
            end = time.time()
            total_time = end - start
            print('total_time: ',total_time)
            epoch_time+=total_time
            train_num+=1
    print(f'total training time for 2 epoch: {total_time}')
    print(f'total training time for 1 epoch: {total_time/2}')
    print(f'Time for one nminibatch: {total_time/train_num}')
        
        # #test
        # test_total = 0.0
        # correct_total = 0
        # best_acc = 0
        # with torch.no_grad():
        #     for index, data in enumerate(test_loader):
        #         print(f'test: total length: {length_test}, index: {index}')
        #         model.eval()
        #         img, label = data
        #         img, label = img.to(device),label.to(device)
        #         output = model(img)
        #         test_loss = loss_fun(output,label)
        #         test_total += test_loss.item()
        #         _, predicted = output.max(1)
        #         correct = predicted.eq(label).sum().item()
        #         print('correct: ',correct)
        #         correct_total+=correct
        #         if correct/label.size(0)*100 > best_acc:
        #             best_acc = correct/label.size(0)*100
        #             best_acc_index = index
        #             best_acc_loss = loss.item()
        #         print(f'test_epoch:{epoch}index:{index}, loss: {test_loss.item():.3f}, ACC: {correct/label.size(0)*100:.3f}%({correct}/{label.size(0)})')
        # average_loss = test_total/index
        # print(f'test_{epoch}_total:, loss: {test_total/index:.3f}, ACC: {correct_total/(label.size(0)*length_test)*100:.3f}%({correct_total}/{label.size(0)*length_test})')
        # print(f'Best accuracy index: {best_acc_index}, best_acc: {best_acc}%, loss: {best_acc_loss}')
        # print('\n')




