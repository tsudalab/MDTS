import mdts
import numpy as np
import itertools



### Thermal conductivity for (Si:Ge=1:1) alloy with 16 atoms in the interfacial structure
DB_16=np.load('./DB_16.npy', allow_pickle=True).item()
print(max(DB_16.values()))
print(len(DB_16))

### The get_reward simulates the structure: takes a python list represents the structure and return the reward
def get_reward(struct):
    s = ''.join(str(x) for x in struct)
    if s in DB_16.keys():
        cond = DB_16[s]
    else:
        print ("error")
    return cond


### Initialize the tree with the following parameters
### no_positions: number of positions in each structure. For example, 16 atoms.
### atom_types: atom types. For example, types of atoms: 0 for Si and 1 for Ge
### atom_const: number of each atom type in the structure. For example, 8 atoms Si and 8 atoms Ge. Default is None
### get_reward: the experiment simulation function
### positions_order: define the order to assign atoms to the positions in the structure: "direct", "reverse",
    #shuffle# or a list. Default is "reverse"
### max_flag: if True the algorithm searches for maximum reward, else for minimum
### expand_children : number of children to expand at each node. Default is "1". i.e. expand one child at a time.
### play_out: number of play outs et each node. Default is 1. Please note if you set the parameter use_combo to True,
    #play_out can not be 1
## play_out_selection: when performing multiple playouts, best or mean is returned. Deafualt is best
### space: numpy ndarray representing the candidates space. Default is None. If specified the "no_positions",
    # "atom_types", and "atom_const" parameters will be ignored and there values will be taken from the data.
    # This is a slower option, not recommended unless there are complex constraints on the structures
    # data needs to be assigned if you want to use the option use_combo=True
### ucb: it can be either "mean" or "best", it represents taking either average or best ucb score for Monte Carlo tree
    # search. Default is "mean"
### use_combo: weather to use Bayesian optimisation or not in combination with Monte Carlo tree search.
    # COMBO package is used to engineer the palyout instead of random selection.
### combo_play_out: total number of candidates to be examind by COMBO.
### combo_init_random: the initial random selection for Bayesian optimisation. Default is 1
### combo_step: the interval for Bayesian optimisation to perfrom hyperparameter optimization. Default is 1
### combo_lvl: the level of the tree at which start to apply Bayesian optimisation. Default is 1 (apply at all levels)

myTree=mdts.Tree(no_positions=16, atom_types=[0,1], atom_const=[8,8], get_reward=get_reward, positions_order=list(range(16)),
                max_flag=True,expand_children=2, play_out=1, play_out_selection="best", space=None, candidate_pool_size=100,
                 ucb="mean", use_combo=True, combo_play_out=20, combo_init_random=5, combo_step=5, combo_lvl=5)

### Start the search for certain number of candidates and returns an object of type Result contains the result of the search
res=myTree.search(display=True,no_candidates=500)

# ### Optimal reward
print (res.optimal_fx)

# ### List of optimal candidates
print (res.optimal_candidate)

# ### List of tuples with the candidates examined and their rewards
print (res.checked_candidates_DS)

# ### Number of examined candidates
print (res.checked_candidates_size)

### List of chosen candidates in order
print (res.checked_candidates)

### List of simulated candidates rewards in order
print (res.fx)

### List of current best reward
print (res.best_fx)

### Maximum depth reached
print (res.max_depth_reached)

### Number of nodes constructed
print (res.no_nodes)

### Average visit per node
print (res.avg_node_visit)

### Save results
res.save('results.npz')

### Load results
new_res=mdts.Result()
new_res.load('results.npz')
print (new_res.optimal_fx)


### The tree can be saved
mdts.save_tree(myTree,'Tree_file')
del myTree

### Load the tree and continue the search
myNewTree = mdts.load_tree('Tree_file')
newTree_res=myNewTree.search(display=True, no_candidates=600)
print (newTree_res.checked_candidates_size)
