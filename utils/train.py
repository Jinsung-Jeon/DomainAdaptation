# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:43:06 2020

@author: Jinsung
"""

import torch
from utils.get_mmd import get_mmd

def test(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    alpha = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs, _ = model(inputs, alpha)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    
    return 1 - correct/total

def train(args, net, ext, sstasks, criterion_cls, criterion_domain, optimizer_cls, scheduler_cls, sc_tr_loader, sc_te_loader, tg_te_loader):
    net.train() 
    for sstask in sstasks:
        sstask.head.train()
        sstask.scheduler.step()
    n_epoch = 0
    n_epoch += 1
    epoch_stats = []
    for batch_idx, (sc_tr_inputs, sc_tr_labels) in enumerate(sc_tr_loader):
        for sstask in sstasks:
            sstask.train_batch()

        sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
        domain_label = torch.zeros(args.batch_size)
        domain_label = domain_label.lond().cuda()
        optimizer_cls.zero_grad()

        p = float(batch_idx + epoch * len(sc_tr_loader)) / n_epoch / len(sc_tr_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        outputs_cls, domain_output = net(sc_tr_inputs, alpha)
        loss_cls = criterion_cls(outputs_cls, sc_tr_labels)
        loss_domain = criterion_domain(domain_output, domain_label)

        data_target_iter = iter(tg_te_loader)
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)
        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size).cdua()
        t_img.cuda()

        domain_label = torch.ones(batch_size)
        domain_labels = domain_label.long().cdua()

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = net(input_img, alpha)
        err_t_domain = loss_domain(domain_output, domain_labels)
        err = err_t_domain + loss_cls + loss_domain
        err.backward()
        optimizer_cls.step()
        
        if batch_idx % args.num_batches_per_test == 0:
            sc_te_err = test(sc_te_loader, net)
            tg_te_err = test(tg_te_loader, net)
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