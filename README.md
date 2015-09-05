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
        
