Usage
=====


After installation:

- import mdts package into your code ::

	import mots

- implement the get_reward (struct) function: This function evaluates the solution, that is the material structure, usually by a simulation function. It takes a python list representing the structure and returns the reward, usually the material property to be optimized. If no reward shall be return, use the **False** value as an output.
- set the search tree object (mdts.Tree): This object will set the parameters for constructing the shallow tree. The parameters are as follows:

	+ no_positions: number of positions in each structure to be set. For example, 16 positions.
	+ atom_types: types of atoms. For example, 0 for Si and 1 for Ge
	+ atom_const: number of each atom type in the structure. For example, 8 atoms Si and 8 atoms Ge. NONE indicates no constraints on number of that atom. Default is None
	+ get_reward: the simulation function to evaluate a solution.
	+ positions_order: define the order to assign atoms to the positions in the structure: "direct", "reverse","shuffle" or a list. Default is "reverse"
	+ max_flag: if True the algorithm searches for maximum reward, else for minimum
	+ expand_children : number of children to expand at each node. Default is "1". i.e. expand one child at a time.
	+ play_out: number of playouts to be performed when creating a new node. Default is 1. Please note if you set the parameter use_combo to True, play_out can not be 1
	+ play_out_selection: when performing multiple playouts, best or mean is returned. Default is best
	+ space: numpy ndarray representing the candidates space if available or enumerable. If the space is not available or too expensive to enumerate, use NONE. Default is None. Note, if you specified the space, the "no_positions", "atom_types", and "atom_const" parameters will be ignored and there values will be taken from the ndarray of the space.
	+ candidate_pool_size: the pool size that will be generated for the Bayesian rollout. It has to be set if you set use_combo=True, otherwise, it can be ignored.
	+ ucb: it can be either "mean" or "best", it represents taking either average or best ucb score for Monte Carlo tree search. Default is "mean"
	+ use_combo: weather to use Bayesian optimisation package (COMBO) or not in combination with Monte Carlo tree search to perform the rollout. If it is set to False, random rollout is performed.
	+ combo_play_out: total number of candidates to be examined by COMBO while performing the rollout operation.
	+ combo_init_random: the initial random selection for Bayesian optimisation. Default is 1
	+ combo_step: the interval for Bayesian optimisation to perform hyper-parameter optimization. Default is 1
::

	myTree=mdts.Tree(no_positions=16, atom_types=[0,1], atom_const=[8,8], get_reward=get_reward, positions_order=range(16),
                max_flag=True, expand_children=2, play_out=5, play_out_selection="best", space=None, candidate_pool_size=100,
                 ucb="mean", use_combo=True, combo_play_out=20, combo_init_random=5, combo_step=5)
- start the search by defining the required number of iterations. Search function will return an object of the class Result::

	res=myTree.search(display=True,no_candidates=500)

- search results can be obtained using the following properties of the class Result:

	+ Optimal reward::

		print res.optimal_fx

	+ List of optimal candidates::

		print res.optimal_candidate

	+ List of tuples with the candidates examined and their rewards::

		print res.checked_candidates_DS

	+ Number of examined candidates::

		print res.checked_candidates_size

	+ List of chosen candidates in order::

		print res.checked_candidates

	+ List of simulated candidates rewards in order::

		print res.fx

	+ List of current best reward::

		print res.best_fx

	+ Maximum depth reached::

		print res.max_depth_reached

	+ Number of nodes constructed::

		print res.no_nodes

	+ Average visit per node::

		print res.avg_node_visit

- save and load results

	+ Save results::

		res.save('results.npz')

	+Load results::

		new_res=mdts.Result()
		new_res.load('results.npz')
		print new_res.optimal_fx


- The tree can be saved::

	mdts.save_tree(myTree,'Tree_file')
	del myTree

- Load the tree and continue the search from where you left it::

	myNewTree = mdts.load_tree('Tree_file')
	newTree_res=myNewTree.search(display=True, no_candidates=600)
	print newTree_res.checked_candidates_size

