"""
ResNet Architecture for computing the representations.
"""
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



class modelA(nn.Module):
    def __init__(self, **param):
        super().__init__()
        self.number_features = param['number_features']
        self.embed_dim = param['embed_dim']
        self.number_labels = param['number_labels']
        self.patch_size = param['patch_size']

        self.bn1 = nn.BatchNorm2d(self.number_features)
        self.conv1 = nn.Conv2d(self.number_features, 32, kernel_size=1, stride=1, bias=False)
        self.cord1 = CoordAtt(32, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, bias=False)
        sz = 2*self.patch_size + 1 - 2*(3-1)

        self.bn4 = nn.BatchNorm1d(16*sz*sz)
        self.fc = nn.Linear(16*sz*sz, self.embed_dim)

    def forward_rep(self, x):
        out = F.relu(self.conv1(self.cord1(self.bn1(x))))
        out = F.relu(self.conv2(self.bn2(out)))
        out = F.relu(self.conv3(self.bn3(out)))
        out = F.relu(self.cord1(out))
        out = torch.reshape(out, (len(out), -1))
        out = self.fc(self.bn4(out))
        return out

    def forward(self, x):
        out = self.forward_rep(x)
        return out
