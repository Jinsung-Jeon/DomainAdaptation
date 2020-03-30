# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:50:26 2020

@author: Jinsung
"""

import os

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
