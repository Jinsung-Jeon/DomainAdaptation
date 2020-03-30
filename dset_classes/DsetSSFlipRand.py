# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:47:39 2020

@author: Jinsung
"""

import torch
import torch.utils.data
import numpy as np

def flip_ver(x):
    return x.flip(1)

def flip_hor(x):
    return x.flip(2)

class Flip(torch.utils.data.Dataset):
    def __init__(self, dset, digit=False):
        self.dset = dset
        if digit:
            self.label_max = 3
        else:
            self.label_max = 2
            
    def __getitem__(self, index):
        image = self.dset[index]
        label = np.random.randint(self.label_max)
        
        if label == 1:
            image = flip_ver(image)
        else:
            image = flip_hor(image)
            
        return image, label
    
    def __len__(self):
        return len(self.dset)
    
