# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from pybrain.tools.customxml.networkreader import NetworkReader
import sys
sys.path.insert(0, "..")
from compartmentalize import *
from read_classifier import *

def decode(classifications, max_length, max_pval):
    """
    Author: Xander
    Given a list of the classifications
    for every allowed starting frame, this
    function attempts to find which stars
    have exoplanets.
    """
    l = len(lightcurve)
    max_length = min(max_length, l)
    f = open("heights.txt")
    lines = f.readlines()
    f.close()
    barmap = [None, None]
    for line in lines:
        height = float(line[:-1].split(":")[1])
        barmap.append(height)
    slope = (barmap[-1] - barmap[-21]) / 20
    while max_length > len(barmap) - 1:
        barmap.append(barmap[-1] + slope)
    min_size = max(2, int(2 / max_pval))
    memovalues = [None]*l + [([l], 0)]
    memovalues[l - min_size + 1:-1] = [([], float("inf"))]*(min_size - 1)
    i = l - min_size
    while i >= 0:
        minimum = min(lightcurve[i:i + min_size])
        maximum = max(lightcurve[i:i + min_size])
        best_badness = float("inf")
        best_partitioning = []
        for j in range(i + min_size, min(l + 1, i + max_length + 1)):
            new_badness = (j - i) * (maximum - minimum) / barmap[j - i]
            found_partitioning, found_badness = memovalues[j]
            if new_badness + found_badness < best_badness:
                best_badness = new_badness + found_badness
                best_partitioning = [i] + found_partitioning
            if j < l and lightcurve[j] > maximum:
                maximum = lightcurve[j]
            elif j < l and lightcurve[j] < minimum:
                minimum = lightcurve[j]
        memovalues[i] = (best_partitioning, best_badness)
        i -= 1
    return memovalues[0]

def detect(lightcurves):
    """
    Author: Xander
    This function hunts for exoplanet
    transits in the lightcurves provided.
    All frames must be taken at equal
    intervals of time, and the number
    of lightcurves must be the same as
    the number of lightcurves to be input 
    in the neural net.
    """
    message_file = open("detect_info.txt")
    lines = message_file.readlines()
    message_file.close()
    stars = int(lines[0][:-1].split(" ")[1])
    points = int(lines[1][:-1].split(" ")[1])
    step = int(lines[2][:-1].split(" ")[1])
    bin_size = int(lines[3][:-1].split(" ")[1])
    dynamic = int(lines[4][:-1].split(" ")[1])
    if dynamic:
        partitionings = [compartmentalize(lightcurve, max_length=step)[0] for lightcurve in lightcurves]
    else:
        partitionings = [range(0, step * points, bin_size) + [step * points] for lightcurve in lightcurves]
    maxs = [[max(lightcurve[partitionings[i]: partitionings[i + 1]]) for i in range(len(partitionings - 1))]
                                                                     for lightcurve in lightcurves]
    mins = [[min(lightcurve[partitionings[i]: partitionings[i + 1]]) for i in range(len(partitionings - 1))]
                                                                     for lightcurve in lightcurves]
    classifications = []
    for i in range(len(lightcurves[0]) - step * (points - 1)):
        inpt = []
        for o in range(stars):
            for num, point in enumerate(partitionings[o][:-1]):
                next_point = partitionings[o][num + 1]
                if ((next_point - 1 - i) // step > (point - 1 - i) // step
                and (point - i) <= (points - 1)*step):
                    if (point - i) % step > 0:
                        start = step * ((point - i) // step + 1) + i
                    else: # (point - i) % step == 0
                        start = point
                    for i in range(start, next_point, transit.generate_step):
                        inpt += [maxs[o][num], mins[o][num]]
        classification = read_classifier(inpt)
        classifications.append(classification)
    return decode(classifications)