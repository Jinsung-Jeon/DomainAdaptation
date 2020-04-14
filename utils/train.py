# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:43:06 2020

@author: Jinsung
"""
import random
import torch
from utils.get_mmd import get_mmd
import numpy as np
from utils.loss import loss_fn_kd
import pdb
from utils.misc import get_dummy, make_data_loader, get_inf_iterator, guess_pseudo_labels, make_variable
from torch.utils.data import ConcatDataset

def test(dataloader, model):
    model.eval()
    correct_cls = 0
    correct_domain = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs_cls = model(inputs)
            #outputs_cls, outputs_domain = model(inputs)
        _, predicted_cls = outputs_cls.max(1)
        #_, predicted_domain = outputs_domain.max(1)
        total += labels.size(0)
        correct_cls += predicted_cls.eq(labels).sum().item()
        #correct_domain += predicted_domain.eq(1).sum().item()
    model.train()
    
    return 1 - correct_cls/total#, 1 - correct_domain/total

def test_d(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs, outputs_domain = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    model.train()
    return 1 - correct / total

def train(args, net, ext, sstasks, criterion_cls, optimizer_cls, scheduler_cls, sc_tr_loader, sc_te_loader, tg_tr_loader, tg_te_loader):
    net.train()
    for sstask in sstasks:
        sstask.head.train()
        sstask.scheduler.step()

    epoch_stats = []
    for batch_idx, (sc_tr_inputs, sc_tr_labels) in enumerate(sc_tr_loader):
        for sstask in sstasks:
            sstask.train_batch()
        #source domain prepare
        sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
        #domain_label = torch.zeros(len(sc_tr_inputs))
        #domain_labels = torch.ones(len(sc_tr_inputs))
        #domain = torch.stack([domain_label, domain_labels], 1)
        #domain_label = domain_label.long().cuda()
        #domain_label = domain.cuda()
        optimizer_cls.zero_grad()

        #pdb.set_trace()
        #source domain train
        outputs_cls, domain_output = net(sc_tr_inputs)
        loss_cls = criterion_cls(outputs_cls, sc_tr_labels)
        #loss_domain = loss_fn_kd(domain_output, domain_label, args).cuda()
        loss_cls.backward()
        #target domain prepare
        #tg_tr_inputs = tg_tr_inputs.cuda()
        #domain_label = torch.ones(len(tg_tr_inputs))
        #domain_labels = torch.zeros(len(tg_tr_inputs))
        #domain = torch.stack([domain_label, domain_labels], 1)
        #domain_label = domain_label.long().cuda()
        #domain_label = domain.cuda()

        #target train
        #_, domain_output = net(tg_tr_inputs)
        #err_t_domain = loss_fn_kd(domain_output, domain_label, args).cuda()

        #err = err_t_domain + loss_cls + loss_domain

        #err.backward()
        optimizer_cls.step()

        #optimizer_cls.zero_grad()
        #tg_tr_inputs = tg_tr_inputs.cuda()
        #_, domain_output = net(tg_tr_inputs)

        #domain_label = torch.zeros(len(tg_tr_inputs))
        #domain_labels = torch.ones(len(tg_tr_inputs))
        #domain = torch.stack([domain_label, domain_labels], 1)
        #domain_label = domain_label.long().cuda()
        #domain_label = domain.cuda()

        #loss_tgt = loss_fn_kd(domain_output, domain_label, args).cuda()
        #loss_tgt. backward()
        #optimizer_cls.step()
        if batch_idx == len(sc_tr_loader)-1:
        #if batch_idx % args.num_batches_per_test == 0:
            sc_te_err = test_d(sc_te_loader, net)
            tg_te_err = test_d(tg_te_loader, net)
            mmd = get_mmd(sc_te_loader, tg_te_loader, ext)

            us_te_err_av = []
            for sstask in sstasks:
                err_av, err_sc, err_tg = sstask.test()
                us_te_err_av.append(err_av)
            epoch_stats.append((batch_idx, len(sc_tr_loader), mmd, tg_te_err, sc_te_err, us_te_err_av,loss_cls))
            display = ('Iteration %d/%d:' %(batch_idx, len(sc_tr_loader))).ljust(24)
            display += '%.2f\t%.2f\t\t%.2f\t\t%.2f\t\t' %(mmd, tg_te_err*100, sc_te_err*100, loss_cls*100)
            for err in us_te_err_av:
                display += '%.2f\t'%(err*100)
            print(display)
    return epoch_stats

def labeling(args, model, tg_tr_loader):
    model.eval()
    out_F_1_total = None
    inputs_idx = None
    for batch_idx, (inputs, _) in enumerate(tg_tr_loader):
        with torch.no_grad():
            outputs_cls, outputs_domain = model(inputs)
        if batch_idx == 0:
            out_F_1_total = outputs_cls.cpu()
            inputs_idx = inputs.cpu()
        else:
            out_F_1_total = torch.cat((out_F_1_total, outputs_cls.cpu()),0)
            inputs_idx = torch.cat((inputs_idx, inputs.cpu()),0)
    excerpt, pseudo_labels, inputs_z = guess_pseudo_labels(args, out_F_1_total, inputs_idx)

    return excerpt, pseudo_labels, inputs_z

def train_d(args, net, ext, sstasks, criterion_cls, optimizer_cls, sc_tr_loader, sc_tr_dataset,sc_te_loader, tg_tr_dataset, tg_te_loader,excerpt, pseudo_labels, input_z):
    target_dataset_labelled = get_dummy(args, tg_tr_dataset, excerpt, pseudo_labels, input_z, get_dataset=True)
    sc_tr_dataset = random.sample(list(sc_tr_dataset), len(input_z))
    merged_dataset = ConcatDataset([sc_tr_dataset, target_dataset_labelled])

    net.train()
    print("pseudo label %.2f" %len(input_z))

    merged_dataloader = make_data_loader(merged_dataset)
    target_dataloader_labelled = get_inf_iterator(make_data_loader(target_dataset_labelled))
    epoch_stats = []
    for batch_idx, (images, labels) in enumerate(next(target_dataset_labelled)):
    #for batch_idx, (images, labels) in enumerate(merged_dataloader):
        #images_tgt, labels_tgt = next(target_dataloader_labelled)

        images = make_variable(images)
        labels = make_variable(labels)
        #images_tgt = make_variable(images_tgt)
        #labels_tgt = make_variable(labels_tgt)

        optimizer_cls.zero_grad()

        #output_cls, _ = net(images)
        output_cls_tgt, _ = net(images_tgt)
        loss_cls = criterion_cls(output_cls_tgt, labels)
        #loss_domain = criterion_cls(output_cls_tgt, labels_tgt)
        loss_cls.backward()
        #err = loss_cls + loss_domain
        #err.backward()
        optimizer_cls.step()

        if batch_idx == len(merged_dataloader)-1:
        #if batch_idx % args.num_batches_per_test == 0:
            sc_te_err = test_d(sc_te_loader, net)
            tg_te_err = test_d(tg_te_loader, net)
            mmd = get_mmd(sc_te_loader, tg_te_loader, ext)

            us_te_err_av = []
            for sstask in sstasks:
                err_av, err_sc, err_tg = sstask.test()
                us_te_err_av.append(err_av)
            epoch_stats.append((batch_idx, len(merged_dataloader), mmd, tg_te_err, sc_te_err, us_te_err_av,loss_cls))
            display = ('Iteration %d/%d:' % (batch_idx, len(sc_tr_loader))).ljust(30)
            display += '%.2f\t%.2f\t\t%.2f\t\t%.2f\t\t' % (mmd, tg_te_err * 100, sc_te_err * 100, loss_cls*100)
            for err in us_te_err_av:
                display += '%.2f\t' % (err * 100)
            print(display)
    return epoch_stats