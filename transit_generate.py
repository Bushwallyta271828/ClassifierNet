# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from numpy.random import normal
from numpy.random import random as uniform
from compartmentalize import compartmentalize

default_generate_step = 100
default_generate_stars = 5
default_generate_points = 10
default_generate_alpha_mean = 0.9
default_generate_alpha_std = 0.1
default_generate_beta_mean = 500
default_generate_beta_std = 20
default_generate_O_mean = 1000
default_generate_O_std = 100
default_generate_O_cutoff = 300
default_generate_planet_frac = 0.33
default_generate_planet_bordertime = 10
default_generate_epsilon_std = 200

class Transit:
    def __init__(self,
                 generate_step             =default_generate_step,
                 generate_stars            =default_generate_stars,
                 generate_points           =default_generate_points,
                 generate_alpha_mean       =default_generate_alpha_mean,
                 generate_alpha_std        =default_generate_alpha_std,
                 generate_beta_mean        =default_generate_beta_mean,
                 generate_beta_std         =default_generate_beta_std,
                 generate_O_mean           =default_generate_O_mean,
                 generate_O_std            =default_generate_O_std,
                 generate_O_cutoff         =default_generate_O_cutoff,
                 generate_planet_frac      =default_generate_planet_frac,
                 generate_planet_bordertime=default_generate_planet_bordertime,
                 generate_epsilon_std      =default_generate_epsilon_std):
        """
        Author: Xander
        This class stores all
        the optional keyword
        arguments for the generate()
        function.
        """
        self.generate_step              = generate_step
        self.generate_stars             = generate_stars
        self.generate_points            = generate_points
        self.generate_alpha_mean        = generate_alpha_mean
        self.generate_alpha_std         = generate_alpha_std
        self.generate_beta_mean         = generate_beta_mean
        self.generate_beta_std          = generate_beta_std
        self.generate_O_mean            = generate_O_mean
        self.generate_O_std             = generate_O_std
        self.generate_O_cutoff          = generate_O_cutoff
        self.generate_planet_frac       = generate_planet_frac
        self.generate_planet_bordertime = generate_planet_bordertime
        self.generate_epsilon_std       = generate_epsilon_std 
        
    def __str__(self):
        """
        Author: Xander
        """
        msg  = "Transit object:\n"
        msg += "     generate_step              = " + str(self.generate_step) + "\n"
        msg += "     generate_stars             = " + str(self.generate_stars) + "\n"
        msg += "     generate_points            = " + str(self.generate_points) + "\n"
        msg += "     generate_alpha_mean        = " + str(self.generate_alpha_mean) + "\n"
        msg += "     generate_alpha_std         = " + str(self.generate_alpha_std) + "\n"
        msg += "     generate_beta_mean         = " + str(self.generate_beta_mean) + "\n"
        msg += "     generate_beta_std          = " + str(self.generate_beta_std) + "\n"
        msg += "     generate_O_mean            = " + str(self.generate_O_mean) + "\n"
        msg += "     generate_O_std             = " + str(self.generate_O_std) + "\n"
        msg += "     generate_O_cutoff          = " + str(self.generate_O_cutoff) + "\n"
        msg += "     generate_planet_frac       = " + str(self.generate_planet_frac) + "\n"
        msg += "     generate_planet_bordertime = " + str(self.generate_planet_bordertime) + "\n"
        msg += "     generate_epsilon_std       = " + str(self.generate_epsilon_std) + "\n"
        return msg
        
default_transit = Transit()

def generate(transit=default_transit):
    """
    Author: Xander
    This function creates the input, output pairs
    required for the neural net to work. 
    """
    alpha = normal(0.9, 0.1, (1000,))
    alpha[alpha > 1] = 1
    beta = normal(500, 20, (1000,))
    inpt = []
    outp = []
    for o in range(5):
        O = normal(1000, 100, (1000,))
        O[O < 300] = 300
        if uniform() < 0.5:
            outp += [False]
        else:
            center = randint(0, 1000)
            spread = randint(50, 500)
            start = center - spread
            stop = center + spread
            span = arange(0, 1000)
            damped_mask = (span > start)*(span < stop)
            O[damped_mask] *= 1 - 0.33
            for x in range(0, 20):
                i = start - x
                if i in span:
                    O[i] *= 1 - 0.33*(1/pi * arccos((10 - x) / 10)
                         - (1 / (2*pi)) * (10 - x) * sqrt((20 - x) * x))
            for x in range(0, 20):
                i = stop + x
                if i in span:
                    O[i] *= 1 - 0.33*(1/pi * arccos((10 - x) / 10)
                         - (1 / (2*pi)) * (10 - x) * sqrt((20 - x) * x))
            outp += [True]
        epsilon = normal(0, 200, (1000,))
        epsilon[-epsilon > O] = 0
        I = alpha * (O + epsilon + beta)
        I /= average(I)
        partitioning = compartmentalize(I)[0]
        for num, point in enumerate(partitioning[:-1]):
            next_point = partitioning[num + 1]
            if (next_point - 1) // 100 > (point - 1) // 100:
                block = I[point: next_point]
                maximum, minimum = max(block), min(block)
                if point % 100 > 0:
                    start = 100 * (point // 100 + 1)
                else:  # point % 100 == 0
                    start = point
                for i in range(start, next_point, 100):
                    inpt += [maximum, minimum]
    return (inpt, outp)
