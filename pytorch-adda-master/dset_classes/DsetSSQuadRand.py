# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:57:33 2020

@author: Jinsung
"""

import torch
import torch.utils.data
import numpy as np

class Quad(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset
        
    def __getitem__(self, index):
        image = self.dset[index]
        label = np.random.randint(4)
        
        horstr = image.size(1) // 2  #
        verstr = image.size(2) // 2
        horlab = label // 2
        verlab = label % 2
        
        image = image[:, horlab*horstr:(horlab+1)*horstr, verlab*berstr:(verlab+1)*verstr,]
        return image, label
    
    def __len__(self):
        return len(self.dset)
        
