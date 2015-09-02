# -*- coding: utf-8 -*-
from __future__ import division
from random import random

default_stop = 500
default_samples = int(1e3)

def avg_dist_find(n, samples=default_samples):
    """
    Author: Xander
    """
    sum_max_dists = 0
    for sample in range(samples):
        max_dist = 0
        has_exoplanets = [[random(), random(), random(), random(), random()] for i in range(n)]
        for i in range(n):
            for j in range(i):
                diff = sum([has_exoplanets[i][k] != has_exoplanets[j][k] for k in range(5)])
                if diff > max_dist:
                    max_dist = diff
        sum_max_dists += max_dist
    return sum_max_dists / samples        

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
    except: #dists.txt doesn't exist yet.
        start = 2
    f = open("dists.txt", "a")
    for n in range(start, stop + 1):
        f.write(str(n) + ":" + str(avg_dist_find(n)) + "\n")
        print n
    f.close()
    
if __name__=="__main__":
    make_file(stop=100)
