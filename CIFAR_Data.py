import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
#图像数据预处理
transform_train=transforms.Compose([
    #数据增强
    transforms.RandomCrop(32, padding=4),#随机裁剪
    transforms.RandomHorizontalFlip(),#随机翻转
    #基本图像数据处理
    transforms.ToTensor(),#转化为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))#标准化
])

transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))#标准化
])

#下载数据并加载
trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader=DataLoader(trainset,batch_size=128,shuffle=True)
testset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader=DataLoader(testset,batch_size=128,shuffle=False)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

