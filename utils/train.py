# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:43:06 2020

@author: Jinsung
"""

import torch
from utils.get_mmd import get_mmd
import numpy as np

def test(dataloader, model):
    model.eval()
    correct_cls = 0
    correct_domain = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs_cls, outputs_domain = model(inputs)
        _, predicted_cls = outputs_cls.max(1)
        _, predicted_domain = outputs_domain.max(1)
        total += labels.size(0)
        correct_cls += predicted_cls.eq(labels).sum().item()
        correct_domain += predicted_domain.eq(1).sum().item()
    print(correct_cls/total)
    print(correct_domain/toal)
    model.train()
    
    return 1 - correct_cls/total, 1 - correct_domain/total

def test_d(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    print(correct/total)
    return 1 - correct / total

def train(args, net, ext, sstasks, criterion_cls, criterion_domain, optimizer_cls, scheduler_cls, sc_tr_loader, sc_te_loader, tg_tr_loader, tg_te_loader):
    net.train() 
    for sstask in sstasks:
        sstask.head.train()
        sstask.scheduler.step()
    epoch_stats = []
    for batch_idx, ((sc_tr_inputs, sc_tr_labels),(tg_tr_inputs, _)) in enumerate(zip(sc_tr_loader,tg_tr_loader)):
        for sstask in sstasks:
            sstask.train_batch()

        #source domain prepare
        sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
        domain_label = torch.zeros(len(sc_tr_inputs))
        domain_label = domain_label.long().cuda()
        optimizer_cls.zero_grad()

        #source domain train
        outputs_cls, domain_output = net(sc_tr_inputs)
        loss_cls = criterion_cls(outputs_cls, sc_tr_labels)
        loss_domain = criterion_domain(domain_output, domain_label)

        #target domain prepare
        tg_tr_inputs = tg_tr_inputs.cuda()
        domain_label = torch.ones(len(tg_tr_inputs))
        domain_label = domain_label.long().cuda()

        #target train
        _, domain_output = net(tg_tr_inputs)
        err_t_domain = criterion_domain(domain_output, domain_label)

        err = err_t_domain + loss_cls + loss_domain

        print(loss_cls)
        print(err_t_domain)
        print(loss_domain)

        err.backward()
        optimizer_cls.step()
        
        if batch_idx % args.num_batches_per_test == 0:
            sc_te_err, sc_domain_err = test(sc_te_loader, net)
            tg_te_err, tg_domain_err = test(tg_te_loader, net)
            mmd = get_mmd(sc_te_loader, tg_te_loader, ext)
            
            us_te_err_av = []
            for sstask in sstasks:
                err_av, err_sc, err_tg = sstask.test()
                us_te_err_av.append(err_av)
                
            epoch_stats.append((batch_idx, len(sc_tr_loader), mmd, tg_te_err, sc_te_err, us_te_err_av))
            display = ('Iteration %d/%d:' %(batch_idx, len(sc_tr_loader))).ljust(24)
            display += '%.2f\t%.2f\t\t%.2f\t\t' %(mmd, tg_te_err*100, sc_te_err*100)
            for err in us_te_err_av:
                display += '%.2f\t'%(err*100)
            print(display)
    
    return epoch_stats