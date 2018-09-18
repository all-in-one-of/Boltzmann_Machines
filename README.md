#AI: Reinforcement Learning and Boltzmann_Machines

Toying with Reinforcement Learning strategies
Implementation of TicTacToe with different learning methods as
-TD(0)
-n-step TD (Temporal Difference)

Toying with implementation of bi- and tripartite Boltzmann Machines

Aim is to compare a bipartite Boltzmann machine (first simply restricted), to a tripartite boltzmann machine (first some kind of restricted machine).

There for the models are implemented with tensorflow and will be compared it training time, some kind of efficience and generativity. 

The idea is the following: We split the hidden layer of the hidden units in to two groups. Allow full connections between the groups. We can train this RBM first and then train the interconnection between the two hidden layers as seperate RBM.
