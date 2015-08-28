#ClassifierNet
An algorithm to detect exoplanets via the transit method using a neural net.  
aperture.py is currently code that will be pertinent later - it is not used at present.
compartmentalize(...) is the same function I have written in my Dynamic2 repository.
See that for an explanation of this function.
heights.txt and make_heights.py accompany compartmentalize. Again, see my Dynamic2 repository.
transit_generate contains the Transit class and the generat() method. Both of these are explained in \__doc\__ strings.
classifier.py creates the neural network from the input, output pairs of transit_generate. See the \__doc\__ strings again for more detail.
network.xml is the actual network. I have run my code for a few minutes, so it is not a very good classifier. Still, were I to run the code for more data and for longer, it would become much, much better. network_info.txt is a text file containing the performance of network.xml
