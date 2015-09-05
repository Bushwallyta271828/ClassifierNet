#ClassifierNet
This repository contains neural net exoplanet detection algorithms.  
For the code present, the only technique used for detecting an exoplanet is the transit method.  
The main points of the algorithm are:  

    Precomputing phase: generate the classifier
        For a good number of different neural networks:
            Create a network with the appropriate number of inputs / outputs.
            For every interation in (quite a few):
                For every simulated exoplanet transit in (quite a few more):
                    simulate a transit
                For every training epoch in (a lot):
                    train the network on the simulated transits.
                Check if current performance on more data is better than ever before for this network.
                    If so: save the changes to file.
                    If not: go back to the way we had it before this iteration
        (Vaguely, use Boosting to generate one "master classifier")
    Computing phase: detect the exoplanets in the data.
        For every position until the end:
            run the master classifier on the data from that point to some fixed distance ahead
        (Vaguely, partition the data up into "similarly classified" blocks)
        For every block:
            If there are any exoplanets found, remember the stars they belong to.
            
The parts of this diagram that have "(Vaguely, ...)" are incomplete.  
The in-code documentation is currently slim, as the code is changing faster than I have time to correct the descriptions that go along with it. Eventually, I plan to have documentation on par with the quality provided in v1.0 of the program. Still, I ought to mention the theoretical functions of each subdirectory:

    BuildNets - Perform the instructions under "For a good number of different neural networks:"
    Common - A collection of scripts common to more than one subdirectory.
    Detect - Perform the computations under the computing phase.
    UseNets - create and use the "master classifier"
    
A more detailed description will be present in the documentation. This will get better toward the finished product / v2.0. 
