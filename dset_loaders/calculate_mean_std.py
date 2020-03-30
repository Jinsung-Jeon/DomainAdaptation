# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:14:36 2020

@author: Jinsung
"""

import sys
import torchvision
import torchvision.transforms as transforms
import numpy as np

dset_name = sys.argv[1]
dset_dir = sys.argv[2]

if dset_name == "mnist":    
    dataset = torchvision.datasets.MNIST(root=dset_dir, train=True)
    print(list(dataset.train_data.size()))
    print(dataset.train_data.float().mean()/255)
    print(dataset.train_data.float().std()/255)

elif dset_name == "svhn":
    dataset = torchvision.datasets.SVHN(root=dset_dir, download=True, split='train')
    print(dataset.data.shape)
    print(dataset.data.mean(axis=(0,2,3))/255)
    print(dataset.data.std(axis=(0,2,3))/255)