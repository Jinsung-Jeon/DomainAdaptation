# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:04:21 2020

@author: Jinsung
"""

import torch
import torch.utils.data

class ThreeChannels(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset
        
    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label  #3channel로 만들
    
    def __len__(self):
        return len(self.dset)
