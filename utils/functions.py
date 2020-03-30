# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:57:31 2020

@author: Jinsung
"""

from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def foward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.net() * -1
        
        return output, None
