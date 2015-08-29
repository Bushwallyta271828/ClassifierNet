# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from numpy.random import normal
from numpy.random import random as uniform
import sys
sys.path.insert(0, "..")
from compartmentalize import compartmentalize
from keywordclasses import *

def generate(transit=default_transit):
    """
    Author: Xander
    This function creates the input, output pairs
    required for the neural net to work. 
    It takes a transit object and uses
    that transit object's values constantly.
    default_transit is a transit object
    created with all the default values.
    """
    alpha = normal(transit.generate_alpha_mean,
                   transit.generate_alpha_std,
                   (transit.generate_step * transit.generate_points,))
    alpha[alpha > 1] = 1
    beta = normal(transit.generate_beta_mean,
                  transit.generate_beta_std,
                  (transit.generate_step * transit.generate_points,))
    inpt = []
    outp = []
    for o in range(transit.generate_stars):
        O = normal(normal(transit.generate_O_mean_mean,
                          transit.generate_O_mean_std),
                   transit.generate_O_std,
                   (transit.generate_step * transit.generate_points,))
        O[O < transit.generate_O_cutoff] = transit.generate_O_cutoff
        if uniform() < 0.5:
            outp += [False]
        else:
            center = randint(0, transit.generate_step * transit.generate_points)
            spread = randint(transit.generate_step // 2,
                             transit.generate_step * transit.generate_points // 2)
            start = center - spread
            stop = center + spread
            span = arange(0, transit.generate_step * transit.generate_points)
            damped_mask = (span > start)*(span < stop)
            O[damped_mask] *= 1 - transit.generate_planet_frac
            for x in range(0, 2*transit.generate_planet_bordertime):
                i = start + x - 2*transit.generate_planet_bordertime + 1
                if i in span:
                    O[i] *= (1
                           - transit.generate_planet_frac * (1/pi
                                                           * arccos((transit.generate_planet_bordertime
                                                                   - x) / transit.generate_planet_bordertime)
                                                           - (1 / (pi*transit.generate_planet_bordertime)**2)
                                                           * (transit.generate_planet_bordertime - x)
                                                           * sqrt((2*transit.generate_planet_bordertime
                                                                 - x) * x)))
            for x in range(0, 2*transit.generate_planet_bordertime):
                i = stop + 2*transit.generate_planet_bordertime - x - 1
                if i in span:
                    O[i] *= (1
                           - transit.generate_planet_frac * (1/pi
                                                           * arccos((transit.generate_planet_bordertime
                                                                   - x) / transit.generate_planet_bordertime)
                                                           - (1 / (pi*transit.generate_planet_bordertime)**2)
                                                           * (transit.generate_planet_bordertime - x)
                                                           * sqrt((2*transit.generate_planet_bordertime
                                                                 - x) * x)))
            outp += [True]
        epsilon = normal(0,
                         transit.generate_epsilon_std,
                         (transit.generate_step * transit.generate_points,))
        epsilon[-epsilon > O] = 0
        I = alpha * (O + epsilon + beta)
        I /= average(I)
        if transit.generate_dynamic:
            partitioning = compartmentalize(I, max_length=transit.generate_step)[0]
        else:
            partitioning = (range(0,
                                  transit.generate_step * transit.generate_points,
                                  transit.generate_bin_size)
                          + [transit.generate_step * transit.generate_points])
        for num, point in enumerate(partitioning[:-1]):
            next_point = partitioning[num + 1]
            if (next_point - 1) // transit.generate_step > (point - 1) // transit.generate_step:
                block = I[point: next_point]
                maximum, minimum = max(block), min(block)
                if point % transit.generate_step > 0:
                    start = transit.generate_step * (point // transit.generate_step + 1)
                else:  # point % transit.generate_step == 0
                    start = point
                for i in range(start, next_point, transit.generate_step):
                    inpt += [maximum, minimum]
    return (inpt, outp)
