# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *

default_stop = 500
default_samples = int(1e6)

def avg_dist_find(n, samples=default_samples):
    """
    Author: Xander
    """
    data = normal(0, 1, (n, samples))
    return average(amax(data, axis=0)
                 - amin(data, axis=0))
    
def make_file(stop=default_stop):
    """
    Author: Xander
    This method calls avg_dist_find(...) for every n
    in the range [start, stop], including stop, and writes
    the output to dists.txt in the format:
        n:avg_dist_find(n)
    """
    try:
        g = open("dists.txt")
        lines = g.readlines()
        g.close()
        line = lines[-1]
        start = int(line.split(":")[0]) + 1
    except IOError: #dists.txt doesn't exist yet.
        start = 2
    f = open("dists.txt", "a")
    for n in range(start, stop + 1):
        f.write(str(n) + ":" + str(avg_dist_find(n)) + "\n")
        print n
    f.close()
    
if __name__=="__main__":
    make_file(stop=100)