# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:50:26 2020

@author: Jinsung
"""

import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable


def write_to_text(name, content):
    with open(name, 'w') as text_file:
        text_file.write(content)
        
def my_makedir(name):
    try:
        os.mkdir(name)
    except OSError:
        pass
    
def print_args(opt):
    for arg in vars(opt):
        print('%s %s' % (arg, getattr(opt, arg)))
        
def mean(ls):
    return sum(ls) / len(ls)

def print_nparams(model):
    nparams = sum([param.nelement() for param in model.parameters()])
    print('numver of parameters: %d' % (nparams))

def guess_pseudo_labels(args,out_1, inputs_idx):
    out_2 = F.softmax(out_1, dim=1)
    pred_1, _ = torch.max(out_2, 1)
    filtered_idx = torch.nonzero(pred_1 > args.threshold).squeeze()
    _, pred_idx = torch.max(out_1[filtered_idx, :], 1)

    pseudo_labels = pred_idx
    excerpt = out_1[filtered_idx]
    inputs = inputs_idx[filtered_idx]

    return excerpt, pseudo_labels, inputs

class DummyDataset(data.Dataset):
    def __init__(self, args, original_dataset, excerpt, pseudo_labels, input_z):
        super(DummyDataset, self).__init__()
        assert len(excerpt) == pseudo_labels.size(0), "Size of excerpt images({}) and pseudo labels({}) aren't equal".format(len(excerpt), pseudo_labels.size(0))
        assert len(input_z) == pseudo_labels.size(0), "Size of excerpt images({}) and pseudo labels({}) aren't equal".format(len(excerpt), pseudo_labels.size(0))
        self.dataset = original_dataset
        self.excerpt = excerpt
        if args.source == 'cifar10':
            self.pseudo_labels = pseudo_labels.numpy()
        else:
            self.pseudo_labels = pseudo_labels
        self.input_z = input_z

    def __getitem__(self, index):
        #images, _ = self.dataset[self.excerpt[index]]
        return self.input_z[index], self.pseudo_labels[index]

    def __len__(self):
        return len(self.excerpt)

def get_dummy(args, original_dataset, excerpt, pseudo_labels, input_z, get_dataset=False, batch_size=256):
    dummy_dataset = DummyDataset(args, original_dataset, excerpt, pseudo_labels, input_z)
    if get_dataset:
        return dummy_dataset
    else:
        dummy_data_loader = torch.utils.data.DataLoader(dataset=dummy_dataset, batch_size=256, shuffle=True)
        return dummy_data_loader

def make_data_loader(dataset, batch_size=256,shuffle=True, sampler=None):
    """Make dataloader from dataset."""
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler)
    return data_loader

def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels in data_loader:
            yield (images, labels)

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

