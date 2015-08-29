# -*- coding: utf-8 -*-
from __future__ import division
from pylab import *
from numpy import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import sys
sys.path.insert(0, "..")
from transit_generate import *
from keywordclasses import *
    
def execute(nnet, transit=default_transit):
    """
    Author: Xander
    execute(nnet, transit) takes a neural net
    and evaluates it on one randonly selected input, output pair.
    It returns an array containing three items:
        True if nnet correctly classifies the data, and False if not.
        The number of false positives.
        The number of false negatives.
    """
    inpt, output = generate(transit=transit)
    nnet_output = (array(nnet.activate(inpt)) > 0.5)
    false_positives = 0
    false_negatives = 0
    for o in xrange(transit.generate_stars):
        if (not output[o]) and nnet_output[o]:
            false_positives += 1
        elif output[o] and (not nnet_output[o]):
            false_negatives += 1
    return array([false_positives + false_negatives == 0,
                  false_positives,
                  false_negatives])

def message(net, size, trainer=default_trainer, transit=default_transit):
    """
    Author: Xander
    Given a neural net and the number of transits to test on,
    this function creates a message for the user.
    It alo returns the fraction of the time the neural net is correct.
    """
    result = array(map(execute, [net]*size, [transit]*size))
    total = sum(result[:,0])
    false_positives = sum(result[:,1])
    false_negatives = sum(result[:,2])
    msg = str(100 * total / size) + "% of the test data was correctly classified.\n"
    msg += "Of the " + str(false_positives + false_negatives) + " incorrect classifications:\n"
    msg += "    " + str(false_positives) + " were false positives.\n"
    msg += "    " + str(false_negatives) + " were false negatives.\n"
    msg += "The trainer object used was:\n"
    msg += trainer.__str__()
    msg += "The transit object used was:\n"
    msg += transit.__str__()
    return (msg, 100 * total / size)


def train_network(net,
                  best_fraction,
                  trainer=default_trainer,
                  transit=default_transit):
    """
    Author: Xander
    This function performs the common grunt-work of 
    both build_network() and improve_network().
    """
    print "Building dataset..."
    ds = SupervisedDataSet(2*transit.generate_stars*transit.generate_points,
                           transit.generate_stars)
    for i in xrange(trainer.interval_count):
        print "Generating exoplanet transits..."
        ds.clear()
        for k in xrange(trainer.data_size):
            inpt, output = generate(transit=transit)
            ds.addSample(inpt, output)
        print "Building trainer..."
        network_trainer = BackpropTrainer(net, ds)
        print "Training..."
        for j in xrange(trainer.interval_size):
            msg = "Iteration"
            msg += " "*(len(str(trainer.interval_count*trainer.interval_size))
                      - len(str(trainer.interval_size*i + j + 1)) + 1)
            msg += str(trainer.interval_size*i + j + 1)
            msg += " of " + str(trainer.interval_count * trainer.interval_size)
            msg += ": error = "
            msg += str(network_trainer.train())
            print msg
        if i != trainer.interval_count - 1:
            print "Creating interval report..."
            report = message(net,
                             trainer.interval_check,
                             trainer=trainer,
                             transit=transit)
            print report[0][:-1]
            if report[1] > best_fraction:
                best_fraction = report[1]
                print "This interval was helpful and will be saved."
                print "Saving..."
                NetworkWriter.writeToFile(net, "../network.xml")
                print "Writing info..."
                f = open("../network_info.txt", "w")
                for line in report[0]:
                    f.write(line)
                f.close()
            else:
                print "This interval was not helpful and will be discarded."
                print "Retreiving older version..."
                net = NetworkReader.readFrom("../network.xml")
    print "Creating program report..."
    report = message(net,
                     trainer.check_size,
                     trainer=trainer,
                     transit=transit)
    print report[0][:-1]
    if report[1] > best_fraction:
        best_fraction = report[1]
        print "This interval was helpful and will be saved."
        print "Saving..."
        NetworkWriter.writeToFile(net, "../network.xml")
        print "Writing info..."
        f = open("../network_info.txt", "w")
        for line in report[0]:
            f.write(line)
        f.close()
    else:
        print "This interval was not helpful and will be discarded."
        print "Retreiving older version..."
        net = NetworkReader.readFrom("../network.xml")
        print "Improving older report..."
        better_report = message(net=net,
                                size=trainer.check_size,
                                trainer=trainer,
                                transit=transit)
        print "Writing info..."
        f = open("../network_info.txt", "w")
        for line in better_report[0]:
            f.write(line)
        f.close()

def build_network(hidden_structure=(512, 256, 128, 64, 32, 16, 8),
                  trainer=default_trainer,
                  transit=default_transit):
    """
    Author: Xander
    This function creates a neural net capable of detecting exoplanets in lightcurves.
    It writes the network to network.xml
    The input, output pairs should be of the 
    format generate() generates them in.
    A good rule-of-thumb for telling whether the network detects an exoplanet
    is to see if the output is above 0.5.
    """
    print "Building network..."
    hidden_structure = ((2*transit.generate_stars*transit.generate_points,)
                      + hidden_structure
                      + (transit.generate_stars,))
    net = buildNetwork(*hidden_structure, bias=True)
    best_fraction = 0
    train_network(net, best_fraction, trainer=trainer, transit=transit)
        
def improve_network(trainer=default_trainer, transit=default_transit):
    """
    Author: Xander
    This function improves an existing neural net
    capable of detecting exoplanets in lightcurves.
    It writes the network to network.xml
    The input, output pairs should be of the 
    format generate() generates them in.
    A good rule-of-thumb for telling whether the network detects an exoplanet
    is to see if the output is above 0.5.
    """
    print "Retreiving network..."
    net = NetworkReader.readFrom("../network.xml")
    print "Retreiving current performance..."
    f = open("../network_info.txt")
    first_line = f.readlines()[0]
    best_fraction = float(first_line.split("%")[0])
    f.close()
    train_network(net, best_fraction, trainer=trainer, transit=transit)