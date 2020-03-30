# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:31:47 2020

@author: Jinsung
"""
import torch
import torch.utils.data
import numpy as np

class Domain1(torch.utils.data.Dataset):
    def __init__(self, dset, digit=False):
        self.dset = dset
            
    def __getitem__(self, index):
        image = self.dset[index]
        label = 1

        return image, label
    
    def __len__(self):
        return len(self.dset)
    
