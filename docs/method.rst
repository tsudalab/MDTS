Method
======

Materials design process is often represented as a black-box function f(x) optimization problem.

Monte Carlo tree search (MCTS) is an iterative, guided-random best-first search method that models the search space as a shallow tree. Each node of the tree represents an assignment of an atom of a certain type into a position in the structure. At the beginning, only the root node exists. Within a predetermined number of required experiments, the tree grows gradually in an iterative manner. Each iteration consists of 4 steps: Selection, Expansion, Simulation, and Backpropagation. In the Selection step, the tree is traversed from root to a leaf node following the child with the best score (children are scored using several methods, most commonly Upper Confidence Bound (UCB)). In Expansion step, children are generated under the selected node. The simulation step checks the merit of the new children by evaluating a full solution obtained at each child using experiment or simulation. Finally, the Backpropagation step updates the node information back to the root. A new iteration then begins.

To obtain a full solution from a shallow tree, several strategies can be used. The most simple one is the random completion (filling the rest of the positions randomly), a cheap and less efficient solution.

We use Bayesian learning to obtain full solution from a shallow tree. Bayesian optimization methods maintain a surrogate model of f(x), most commonly, Gaussian process (GP). A pool of candidate is generated where each data point represents a full structure. GP starts with an initial set of randomly selected data points from the pool. GP is updated as more data points are observed. An acquisition function is, then, used to determine where to query f(x) by quantifying how promising a data point is using both predicted value and prediction uncertainty. 

.. image:: /_static/images/hybrid.png
