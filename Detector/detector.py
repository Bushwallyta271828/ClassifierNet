# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from pybrain.tools.customxml.networkreader import NetworkReader
import sys
sys.path.insert(0, "..")
from compartmentalize import *
from read_classifier import *

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
    transit_index = lines.index("Transit object:")
    transit_text = lines[transit_index:]
    transit = Transit()
    transit.reconstruct(transit_text)
    points = transit.generate_points
    step = transit.generate_step
    bin_size = transit.generate_bin_size
    dynamic = transit.generate_dynamic
    if dynamic:
        partitionings = [compartmentalize(lightcurve, max_length=step)[0] for lightcurve in lightcurves]
    else:
        partitionings = [range(0, step * points, bin_size) + [step * points] for lightcurve in lightcurves]
    maxs = [[max(lightcurve[partitionings[i]: partitionings[i + 1]]) for i in range(len(partitionings - 1))]
                                                                     for lightcurve in lightcurves]
    mins = [[min(lightcurve[partitionings[i]: partitionings[i + 1]]) for i in range(len(partitionings - 1))]
                                                                     for lightcurve in lightcurves]
    classifications = []
    for i in range(len(lightcurves[0]) - step * (points - 1)): #Check
        inpt = []
        for o in range(len(lightcurves)):
            for num, point in enumerate(partitionings[o][:-1]):
                next_point = partitionings[o][num + 1]
                if ((next_point - 1 - i) // step > (point - 1 - i) // step
                and (point - i) <= (points - 1)*steps): #Check
                    if (point - i) % transit.generate_step > 0:
                        start = transit.generate_step * ((point - i) // transit.generate_step + 1) + i
                    else:  # (point - i) % transit.generate_step == 0
                        start = point
                    for i in range(start, next_point, transit.generate_step):
                        inpt += [maxs[o][num], mins[o][num]]
        classification = read_classifier(inpt)
        classifications.append(classification)
    