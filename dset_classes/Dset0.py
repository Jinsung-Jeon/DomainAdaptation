# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 00:41:48 2020

@author: Jinsung
"""

import torch
import torch.utils.data
import numpy as np

class Domain0(torch.utils.data.Dataset):
    def __init__(self, dset, digit=False):
        self.dset = dset
            
    def __getitem__(self, index):
        image = self.dset[index]
        label = 0

        return image, label
    
    def __len__(self):
        return len(self.dset)