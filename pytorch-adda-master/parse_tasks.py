# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:45:48 2020

@author: Jinsung
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
import params
from core.pretrain import eval_src as test
from SSTask import SSTask
from SSHead import linear_on_layer3
from dset_classes.DsetNoLabel import DsetNoLabel

def parse_tasks(ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset):
    sstasks = []
    
    if params.rotation:
        print("Task : rotation prediction")
        from dset_classes.DsetSSRotRand import Rotation
        
        digit = False
        if params.src_dataset in ['mnist', 'mnistm', 'svhn', 'svhn_exta', 'usps']:
            print("No rotation 180 for digits!")
            digit = True
            
        su_tr_dataset = Rotation(DsetNoLabel(sc_tr_dataset), digit = digit)
        su_te_dataset = Rotation(DsetNoLabel(sc_te_dataset), digit = digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=4)
        
        tu_tr_dataset = Rotation(DsetNoLabel(tg_tr_dataset), digit = digit)
        tu_te_dataset = Rotation(DsetNoLabel(tg_te_dataset), digit = digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=4)
        
        
        head = linear_on_layer3(4, 2, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), lr = 0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler, su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
        
    if params.quadrant:
        print("Task: quadrant prediction")
        from dset_classes.DsetSSQuadRand import Quad
            
        su_tr_dataset = Quad(DsetNoLabel(sc_tr_dataset))
        su_te_dataset = Quad(DsetNoLabel(sc_te_dataset))
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=4)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=4)
        
        tu_tr_dataset = Quad(DsetNoLabel(tg_tr_dataset))
        tu_te_dataset = Quad(DsetNoLabel(tg_te_dataset))
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=4)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=4)
        
        
        head = linear_on_layer3(4, 2, 4).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), lr = 0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler, su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)
        
    if prams.flip:
        print("Task: flip prediction")
        from dset_classes.DsetSSFlipRand import Flip
            
        digit = False
            
        su_tr_dataset = Flip(DsetNoLabel(sc_tr_dataset), digit = digit)
        su_te_dataset = Flip(DsetNoLabel(sc_te_dataset), digit = digit)
        su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=0)
        su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=0)
        
        tu_tr_dataset = Flip(DsetNoLabel(tg_tr_dataset), digit = digit)
        tu_te_dataset = Flip(DsetNoLabel(tg_te_dataset), digit = digit)
        tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=params.batch_size//2, shuffle=True, num_workers=0)
        tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=params.batch_size//2, shuffle=False, num_workers=0)
        
        
        head = linear_on_layer3(2, 2, 8).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), lr = 0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1, last_epoch=-1)
        sstask = SSTask(ext, head, criterion, optimizer, scheduler, su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
        sstask.assign_test(test)
        sstasks.append(sstask)

    return sstasks
