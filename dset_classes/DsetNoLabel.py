# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:01:22 2020

@author: Jinsung
"""

import torch
import torch.utils.data

class DsetNoLabel(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset
        
    def __getitem__(self, index):
        return self.dset[index][0]
    
    def __len__(self):
        return len(self.dset)