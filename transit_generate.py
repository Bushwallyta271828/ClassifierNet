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
default_generate_O_mean_mean = 1000
default_generate_O_mean_std = 100
default_generate_O_std = 100
default_generate_O_cutoff = 300
default_generate_planet_frac = 0.33
default_generate_planet_bordertime = 10
default_generate_epsilon_std = 200
default_generate_bin_size = 40
default_generate_dynamic = False

class Transit:
    def __init__(self,
                 generate_step             =default_generate_step,
                 generate_stars            =default_generate_stars,
                 generate_points           =default_generate_points,
                 generate_alpha_mean       =default_generate_alpha_mean,
                 generate_alpha_std        =default_generate_alpha_std,
                 generate_beta_mean        =default_generate_beta_mean,
                 generate_beta_std         =default_generate_beta_std,
                 generate_O_mean_mean      =default_generate_O_mean_mean,
                 generate_O_mean_std       =default_generate_O_mean_std,
                 generate_O_std            =default_generate_O_std,
                 generate_O_cutoff         =default_generate_O_cutoff,
                 generate_planet_frac      =default_generate_planet_frac,
                 generate_planet_bordertime=default_generate_planet_bordertime,
                 generate_epsilon_std      =default_generate_epsilon_std,
                 generate_bin_size         =default_generate_bin_size,
                 generate_dynamic          =default_generate_dynamic):
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
        self.generate_O_mean_mean       = generate_O_mean_mean
        self.generate_O_mean_std        = generate_O_mean_std
        self.generate_O_std             = generate_O_std
        self.generate_O_cutoff          = generate_O_cutoff
        self.generate_planet_frac       = generate_planet_frac
        self.generate_planet_bordertime = generate_planet_bordertime
        self.generate_epsilon_std       = generate_epsilon_std 
        self.generate_bin_size          = generate_bin_size
        self.generate_dynamic           = generate_dynamic
        
    def __str__(self):
        """
        Author: Xander
        This is the __str__ method of
        any of my printable classes.
        It just returns a string containing
        a nicely human-readable formatting
        of the values in the transit object.
        """
        msg  = "Transit object:\n"
        msg += "     generate_step              = " + str(self.generate_step) + "\n"
        msg += "     generate_stars             = " + str(self.generate_stars) + "\n"
        msg += "     generate_points            = " + str(self.generate_points) + "\n"
        msg += "     generate_alpha_mean        = " + str(self.generate_alpha_mean) + "\n"
        msg += "     generate_alpha_std         = " + str(self.generate_alpha_std) + "\n"
        msg += "     generate_beta_mean         = " + str(self.generate_beta_mean) + "\n"
        msg += "     generate_beta_std          = " + str(self.generate_beta_std) + "\n"
        msg += "     generate_O_mean_mean       = " + str(self.generate_O_mean_mean) + "\n"
        msg += "     generate_O_mean_std        = " + str(self.generate_O_mean_std) + "\n"
        msg += "     generate_O_std             = " + str(self.generate_O_std) + "\n"
        msg += "     generate_O_cutoff          = " + str(self.generate_O_cutoff) + "\n"
        msg += "     generate_planet_frac       = " + str(self.generate_planet_frac) + "\n"
        msg += "     generate_planet_bordertime = " + str(self.generate_planet_bordertime) + "\n"
        msg += "     generate_epsilon_std       = " + str(self.generate_epsilon_std) + "\n"
        msg += "     generate_bin_size          = " + str(self.generate_bin_size) + "\n"
        msg += "     generate_dynamic           = " + str(self.generate_dynamic) + "\n"
        return msg
        
default_transit = Transit()

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
