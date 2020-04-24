# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:24:38 2020

@author: Jinsung
"""

import torch
import torch.utils.data
import numpy as np

def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)
def tensor_rot_90_digit(x):
    return x.transpose(1,2)

def tensor_rot_180(x):
    return x.flip(2).flip(1)
def tensor_rot_180_digit(x):
    return x.flip(2)

def tensor_rot_270(x):
    return x.transpose(1,2).flip(2)

class Rotation(torch.utils.data.Dataset):
    def __init__(self, dset, digit=False):
        self.dset = dset
        self.digit = digit
        
    def __getitem__(self, index):
        image = self.dset[index]
        label = np.random.randint(4)
        if label == 1:
            if self.digit:
                image = tensor_rot_90_digit(image)
            else:
                image = tensor_rot_90(image)
                
        elif label == 2:
            if self.digit:
                image = tensor_rot_180_digit(image)
            else:
                image = tensor_rot_180(image)
                
        elif label == 3:
            image = tensor_rot_270(image)
        return image, label
    
    def __len__(self):
        return len(self.dset)