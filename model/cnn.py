import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class CNNNet(nn.Module):
    def __init__(self,num_input_channels=3,num_classes=10):
        super(CNNNet, self).__init__()
        #卷积层（32×32×3）
        self.conv1=nn.Conv2d(3,16,3,padding=1)
        #卷积层（16×16×16）
        self.conv2=nn.Conv2d(16,32,3,padding=1)
        #卷积层（8×8×32）
        self.conv3=nn.Conv2d(32,64,3,padding=1)
        #最大池化层
        self.pool=nn.MaxPool2d(2,2)
        #Linear layer(4×4×64->500)
        self.fc1=nn.Linear(4*4*64,500)
        #Linear layer(500->10)
        self.fc2=nn.Linear(500,10)
        #dropout(p=0.3)
        self.dropout=nn.Dropout(0.3)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        #flatten image input
        x=x.view(-1,64*4*4)
        # add dropout layer
        x=self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x=F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        x=self.fc2(x)
        return x































        # add sequence of convolutional and max pooling lay



