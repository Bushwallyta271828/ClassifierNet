# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *

default_generate_step              = 100
default_generate_stars             = 5
default_generate_points            = 10
default_generate_alpha_mean        = 0.9
default_generate_alpha_std         = 0.1
default_generate_beta_mean         = 500
default_generate_beta_std          = 20
default_generate_O_mean_mean       = 1000
default_generate_O_mean_std        = 100
default_generate_O_std             = 100
default_generate_O_cutoff          = 300
default_generate_planet_frac       = 0.33
default_generate_planet_bordertime = 10
default_generate_epsilon_std       = 200
default_generate_bin_size          = 40
default_generate_dynamic           = False

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
        msg += "     generate_step              = " + str(self.generate_step)              + "\n"
        msg += "     generate_stars             = " + str(self.generate_stars)             + "\n"
        msg += "     generate_points            = " + str(self.generate_points)            + "\n"
        msg += "     generate_alpha_mean        = " + str(self.generate_alpha_mean)        + "\n"
        msg += "     generate_alpha_std         = " + str(self.generate_alpha_std)         + "\n"
        msg += "     generate_beta_mean         = " + str(self.generate_beta_mean)         + "\n"
        msg += "     generate_beta_std          = " + str(self.generate_beta_std)          + "\n"
        msg += "     generate_O_mean_mean       = " + str(self.generate_O_mean_mean)       + "\n"
        msg += "     generate_O_mean_std        = " + str(self.generate_O_mean_std)        + "\n"
        msg += "     generate_O_std             = " + str(self.generate_O_std)             + "\n"
        msg += "     generate_O_cutoff          = " + str(self.generate_O_cutoff)          + "\n"
        msg += "     generate_planet_frac       = " + str(self.generate_planet_frac)       + "\n"
        msg += "     generate_planet_bordertime = " + str(self.generate_planet_bordertime) + "\n"
        msg += "     generate_epsilon_std       = " + str(self.generate_epsilon_std)       + "\n"
        msg += "     generate_bin_size          = " + str(self.generate_bin_size)          + "\n"
        msg += "     generate_dynamic           = " + str(self.generate_dynamic)           + "\n"
        return msg
        
    def reconstruct(self, lines):
        self.generate_step              = int(  lines[ 1].split(" = ")[1])
        self.generate_stars             = int(  lines[ 2].split(" = ")[1])
        self.generate_points            = int(  lines[ 3].split(" = ")[1])
        self.generate_alpha_mean        = float(lines[ 4].split(" = ")[1])
        self.generate_alpha_std         = float(lines[ 5].split(" = ")[1])
        self.generate_beta_mean         = int(  lines[ 6].split(" = ")[1])
        self.generate_beta_std          = int(  lines[ 7].split(" = ")[1])
        self.generate_O_mean_mean       = int(  lines[ 8].split(" = ")[1])
        self.generate_O_mean_std        = int(  lines[ 9].split(" = ")[1])
        self.generate_O_std             = int(  lines[10].split(" = ")[1])
        self.generate_O_cutoff          = int(  lines[11].split(" = ")[1])
        self.generate_planet_frac       = float(lines[12].split(" = ")[1])
        self.generate_planet_bordertime = int(  lines[13].split(" = ")[1])
        self.generate_epsilon_std       = int(  lines[14].split(" = ")[1])
        self.generate_bin_size          = int(  lines[15].split(" = ")[1])
        self.generate_dynamic           = bool( lines[16].split(" = ")[1])
        
default_transit = Transit()


default_data_size      = 1000
default_interval_size  = 20
default_interval_check = 1000
default_interval_count = 10
default_check_size     = 10000
        
class Trainer:
    def __init__(self,
                 data_size     =default_data_size,
                 interval_size =default_interval_size,
                 interval_check=default_interval_check,
                 interval_count=default_interval_count,
                 check_size    =default_check_size):
        """
        Author: Xander
        This class serves the same purpose as Transit,
        just for the train_network() function.
        """
        self.data_size      = data_size
        self.interval_size  = interval_size
        self.interval_check = interval_check
        self.interval_count = interval_count
        self.check_size     = check_size
    
    def __str__(self):
        """
        Author: Xander
        This function allows for the printing of Trainer objects.
        """
        desc = "Trainer object:\n"
        desc += "    data_size = "      + str(self.data_size)      + "\n"
        desc += "    interval_size = "  + str(self.interval_size)  + "\n"
        desc += "    interval_check = " + str(self.interval_check) + "\n"
        desc += "    interval_count = " + str(self.interval_count) + "\n"
        desc += "    check_size = "     + str(self.check_size)     + "\n"
        return desc
        
    def reconstruct(self, lines):
        self.data_size      = int(lines[1].split(" = ")[1])
        self.interval_size  = int(lines[2].split(" = ")[1])
        self.interval_check = int(lines[3].split(" = ")[1])
        self.interval_count = int(lines[4].split(" = ")[1])
        self.check_size     = int(lines[5].split(" = ")[1])
        
default_trainer = Trainer()