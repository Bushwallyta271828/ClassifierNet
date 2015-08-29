# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from aperture import *
from pybrain.tools.customxml.networkreader import NetworkReader
import sys
sys.path.insert(0, "../Classifier")
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
    