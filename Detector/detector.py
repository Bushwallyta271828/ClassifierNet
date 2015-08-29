# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from aperture import *
from pybrain.tools.customxml.networkreader import NetworkReader
import sys
sys.path.insert(0, "..")
from keywordclasses import *
from compartmentalize import *

def detect(lightcurves):
    """
    Author: Xander
    This function hunts for exoplanet
    transits in the lightcurves provided.
    All frames must be taken at equal
    intervals of time.
    """
    message_file = open("../Classifier/network_info.txt")
    lines = message_file.readlines()
    message_file.close()
    trainer_index = lines.index("Trainer object:")
    transit_index = lines.index("Transit object:")
    trainer_text = lines[trainer_index: transit_index - 1]
    transit_text = lines[transit_index:]
    trainer = Trainer()
    trainer.reconstruct(trainer_text)
    transit = Transit()
    transit.reconstruct(transit_text)
    