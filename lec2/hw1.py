import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import random
import cv2
import torchvision


class Classdataset(Dataset):
    def __init__(self, train):
        
        if train:
            self.data = torchvision.datasets.CIFAR10(root="./dataset3",train=True,download=True)
        else:
            self.data = torchvision.datasets.CIFAR10(root="./dataset3",train=False,download=True)
        self.transform = transforms.Compose([
            #Mask(),# optoinal
            transforms.Resize([330,330]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self, index):
        
        img,label = self.data[index]
        # img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.names)


