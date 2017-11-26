# -*- coding: utf-8 -*-
# This file is a bit messy. Need to clean this up a little

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

def darknetConv2D(in_channel, out_channel, bn=True):
    if(bn):
        return Chain(
            c  = L.Convolution2D(in_channel, out_channel, ksize=3, pad=1, nobias=True),
            n  = L.BatchNormalization(out_channel, use_beta=False, eps=0.000001),
            b  = L.Bias(shape=[out_channel,]),
        )
    else:
        return Chain(
            c  = L.Convolution2D(in_channel,out_channel, ksize=3, pad=1,nobias=True),
            b  = L.Bias(shape=[out_channel,]),
        )

# Convolution -> ReLU -> Pooling
def CRP(c, h, stride=2, pooling=True):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h), test=True))
    h = F.leaky_relu(h,slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=0)
    return h

class TinyYolo(Chain): # Might need to reduce the no of parameters here
    def __init__(self):
        super(TinyYolo, self).__init__(
            c1 = darknetConv2D(3, 16),
            c2 = darknetConv2D(None, 32),
            c3 = darknetConv2D(None, 64),
            c4 = darknetConv2D(None, 128),
            c5 = darknetConv2D(None, 256),
            c6 = darknetConv2D(None, 512),
            c7 = darknetConv2D(None, 1024),
            c8 = darknetConv2D(None, 1024),
            c9 = darknetConv2D(None, 6, bn=False)
        )
    def __call__(self,x):
       return self.predict(x)

    def predict(self, x):
        h = CRP(self.c1, x)
        h = CRP(self.c2, h)
        h = CRP(self.c3, h)
        h = CRP(self.c4, h)
        h = CRP(self.c5, h)
        h = CRP(self.c6, h, stride=1)
        h = F.get_item(h,(slice(None),slice(None),slice(1,14),slice(1,14))) # x[:,:,0:13,0:13]
        h = CRP(self.c7, h, pooling=False)
        h = CRP(self.c8, h, pooling=False)
        h = self.c9.b(self.c9.c(h)) # no leaky relu, no BN
        return h
    
class PawanNet(Chain): # This is slightly messy. Also why are the no of channels not increasing. WTF happened to information volume conservation?
    def __init__(self):
        super(ConvNet, self).__init__(
            l1=L.Convolution2D(None,32,ksize=(3,3),stride=1,pad=1),
            l2=L.Convolution2D(32,32,ksize=(3,3),stride=1,pad=1),
            l3=L.DilatedConvolution2D(32,32,ksize=(5,5),stride=1,pad=1),
            l4=L.Convolution2D(32,32,ksize=(3,3),stride=1,pad=1),
            l5=L.Convolution2D(32,32,ksize=(3,3),stride=1,pad=1),    
            l6=L.DilatedConvolution2D(32,32,ksize=(7,7),stride=1,pad=1),
            l7=L.Convolution2D(32,32,ksize=(3,3),stride=1,pad=1),
            l8=L.Convolution2D(32,32,ksize=(3,3),stride=1,pad=1),
            l9=L.Convolution2D(None,6,ksize=(3,3),stride=1,pad=1)
            
        )
        
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2= F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
#         h4 = F.max_pooling_2d(h3, 2)
        
        h5 = F.relu(self.l4(h3))
        h6= F.relu(self.l5(h5))
        h7 = F.relu(self.l6(h6))
        h8 = F.relu(self.l7(h7))
        h9 = self.l8(F.sigmoid(h8))
        return self.l9(h9)