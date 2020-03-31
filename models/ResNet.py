# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 23:53:52 2020

@author: Jinsung
"""

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
from utils.functions import ReverseLayerF

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
    def forward(self, x):
        residual = x
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)
        
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual
    
class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn
        
    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)
    
class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, block=BasicBlock, classes=10, channels=3):
        assert(depth - 2) % 6  == 0
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16 * width)
        self.layer2 = self._make_layer(block, 32 * width, stride=2)
        self.layer3 = self._make_layer(block, 64 * width, stride=2)
        self.bn = nn.BatchNorm2d(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * width, classes)
        self.fc2 = nn.Linear(64 * width, 2)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x, alpha):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        domain_output = self.fc2(reverse_feature)
        x = self.fc(x)
        
        return x, domain_output

