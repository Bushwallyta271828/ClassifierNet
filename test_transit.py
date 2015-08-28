# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from transit_generate import *
from matplotlib.pyplot import plot, show

def visualize_one():
    """
    Author: Xander
    """
    colors = ["b", "g", "r", "c", "m"]
    data, out, Is = generate()
    for j in range(1):
        for i in range(10):
            plot([100*i + 5*j, 100*i + 5*j], [data[20*j + 2*i], data[20*j + 2*i + 1]], colors[j], linewidth=5)
    for j in range(1):
        I = Is[j]
        plot(I, colors[j])
    show()
    print out
    
visualize_one()