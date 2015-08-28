# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from numpy.random import normal
from numpy.random import random as uniform
from compartmentalize import compartmentalize

def generate():
    """
    This function creates the input, output pairs
    required for the neural net to work. 
    """
    alpha = normal(0.9, 0.1, (1000,))
    alpha[alpha > 1] = 1
    beta = normal(500, 20, (1000,))
    inpt = []
    outp = []
    for o in range(5):
        if uniform() < 0.5:
            O = normal(1000, 100, (1000,))
            O[O < 300] = 300
            epsilon = normal(0, 200, (1000,))
            epsilon[-epsilon > O] = 0
            I = alpha * (O + epsilon + beta)
            partitioning = compartmentalize(I)[0]
            for num, point in enumerate(partitioning[:-1]):
                next_point = partitioning[num + 1]
                if 