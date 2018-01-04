# Boltzmann_Machines
Toying with implementation of 2nd and 3rd order Boltzmann Machines

Aim is to compare a second order Boltzmann machine (first simply restricted), i.e. a bipartite graph, to a third order boltzmann machine (first some kind of restricted machine), i.e. a three-partite graph.

There for the models are implemented with tensorflow and will be compared it training time, some kind of efficience and generativity. 

The idea is the following: We split the hidden layer of the hidden units in to two groups without connection within the group. Then the weight matrix will become block-diagonal. 

We can train this RBM first and then train the interconnection between the two hidden layers as seperate RBM.
