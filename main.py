# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 23:56:51 2020

@author: Jinsung
"""

#knowledge distillation 추가

from __future__ import print_function
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.utils.data as torchdata

from models.ResNet import ResNetCifar as ResNet
from dset_loaders.prepare_dataset import prepare_dataset

from utils.train import train, labeling, train_d
from utils.parse_tasks import parse_tasks
from utils.SSHead import extractor_from_layer3
from utils.plot_all_epoch_stats import plot_all_epoch_stats
from utils.misc import *
from utils.loss import loss_fn_kd

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True)
################################################################
parser.add_argument('--source', required=True)
parser.add_argument('--target', required=True)
################################################################
parser.add_argument('--nepoch', default=15, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--milestone_1', default=5, type=int)
parser.add_argument('--milestone_2', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_batches_per_test', default=100, type=int)
################################################################
parser.add_argument('--rotation', action='store_true')
parser.add_argument('--lr_rotation', default=0.1, type=float)
parser.add_argument('--quadrant', action='store_true')
parser.add_argument('--lr_quadrant', default=0.1, type=float)
parser.add_argument('--flip', action='store_true')
parser.add_argument('--lr_flip', default=0.1, type=float)
parser.add_argument('--domain', action='store_true')
parser.add_argument('--lr_domain', default=0.1, type=float)
parser.add_argument('--threshold', default=0.5, type=float)
################################################################
parser.add_argument('--temperature', default=3, type=int)
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=2, type=int)
parser.add_argument('--outf', default='output/demo')
parser.add_argument('--data_root', default='/data/datasets/')
args = parser.parse_args()
my_makedir(args.outf)
cudnn.benchmark = True


image_size = 32
if (args.source=='usps' and args.target=='mnist') or (args.source=='mnist' and args.target=='usps'):
    channels = 1
else:
    channels = 3

if (args.source=='cifar10' and args.target=='stl10') or (args.source=='stl10' and args.target=='cifar10'):
    classes = 9
else:
    classes = 10


print('==> Building model..')
net = ResNet(args.depth, args.width, classes=classes, channels=channels)
net_d = ResNet(args.depth, args.width, classes=classes, channels=channels)
ext = extractor_from_layer3(net)
net = torch.nn.DataParallel(net)
net_d = torch.nn.DataParallel(net_d)
ext = torch.nn.DataParallel(ext)

net.cuda()
net_d.cuda()
ext.cuda()
    
print('==> Preparing datasets..')
sc_tr_dataset, sc_te_dataset = prepare_dataset(args.source, image_size, channels, path=args.data_root)
sc_tr_loader = torchdata.DataLoader(sc_tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
sc_te_loader = torchdata.DataLoader(sc_te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

tg_tr_dataset, tg_te_dataset = prepare_dataset(args.target, image_size, channels, path=args.data_root)
tg_tr_loader = torchdata.DataLoader(tg_tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
tg_te_loader = torchdata.DataLoader(tg_te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

sstasks = parse_tasks(args, ext, sc_tr_dataset, sc_te_dataset, tg_tr_dataset, tg_te_dataset)
criterion = nn.CrossEntropyLoss().cuda()

parameters = list(net.parameters())
for sstask in sstasks:
    parameters += list(sstask.head.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
    
all_epoch_stats = []

print('==> Running..')
for epoch in range(1, args.nepoch+1):
    print('Source epoch %d/%d lr=%.3f' %(epoch, args.nepoch, optimizer.param_groups[0]['lr']))
    print('Error (%)\t\tmmd\ttarget test\tsource test\tloss\tunsupervised test')
    epoch_stats = train(args, net, ext, sstasks, 
        criterion, optimizer, scheduler, sc_tr_loader, sc_te_loader, tg_tr_loader, tg_te_loader)
    #all_epoch_stats.append(epoch_stats)
    #plot_all_epoch_stats(all_epoch_stats, args.outf)
    if args.method == 'self-supervision_Adapt':
        excerpt, pseudo_labels, input_z = labeling(args, net, tg_tr_loader)
        epoch_stats = train_d(args, net, ext, sstasks, criterion, optimizer, sc_tr_loader, sc_tr_dataset, sc_te_loader, tg_te_dataset, tg_te_loader, excerpt, pseudo_labels, input_z)
    all_epoch_stats.append(epoch_stats)
    torch.save(all_epoch_stats, args.outf + '/loss.pth')
    plot_all_epoch_stats(all_epoch_stats, args.outf)
