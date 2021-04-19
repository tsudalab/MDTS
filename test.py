import mdts
import numpy as np


### Thermal conductivity for (Si:Ge=1:1) alloy with 16 atoms in the interfacial structure
DB_16=np.load('./DB_16.npy', allow_pickle=True).item()
print(len(DB_16))
print(max(DB_16.values()))
print(min(DB_16.values()))


### The get_reward simulates the structure: takes a python list represents the structure and return the reward
def get_reward(struct):
    s = ''.join(str(x) for x in struct)
    if s in DB_16.keys():
        cond = DB_16[s]
    else:
        print ("error")
    return cond

### The constraints to check before simulating the structure (if any): takes a python list represents the structure
### and return True if the constraints applies or False otherwise
### default is constraints=None
def constraints(struct):
    count0=struct.count(0)
    count1=struct.count(1)
    if (count0==count1):
        return True
    else:
        return False



### Initialize the tree with the following parameters
### no_positions: number of positions in each structure. For example, 16 atoms.
### position_values: values to be assigned to each position. For example, 0 for Si and 1 for Ge
### if the position_values are the same for all the position you may input them as a list. If each position in the structre
### has different values to be assigned you may input them as a list of lists
### position_values_const: number of each position_values in the structure. For example, 8 atoms Si and 8 atoms Ge. Default is None
### positions_order: define the order to assign atoms to the positions in the structure: "direct", "reverse",
    #shuffle# or a list. Default is "reverse"
### max_flag: if True the algorithm searches for maximum reward, else for minimum
### get_reward: the experiment simulation function
### constraints: the function to define any constraints to be checked on the structre
### expand_children : number of children to expand at each node. Default is "1". i.e. expand one child at a time.
### play_out: number of play outs et each node. Default is 1. Please note if you set the parameter use_combo to True,
    #play_out can not be 1
## play_out_selection: when performing multiple playouts, best or mean is returned. Deafualt is best
### use_combo: Whether to use Bayesian optimisation or not in combination with Monte Carlo tree search.
    # COMBO package is used to engineer the palyout instead of random selection.
### candidate_pool_size: the size of the candidates pool to be generated so that combo can use it as a space to pick up optimal candidate
### combo_lvl: the level of the tree at which start to apply Bayesian optimisation. Default is 1 (apply at all levels)
### combo_init_random: the initial random selection for Bayesian optimisation. Default is 1
### combo_step: the interval for Bayesian optimisation to perfrom hyperparameter optimization. Default is 1
### combo_play_out: total number of candidates to be examind by COMBO.
### use_PG: Whether to use policy gradient in combination with MCTS or not (you may use both combo and PG together too)
### PG_batch_size: the size of the batch to be used to train the neural network

myTree=mdts.Tree(no_positions=16, position_values=[0,1], position_values_const=[8,8], positions_order=list(range(16)), max_flag=True, get_reward=get_reward, constraints=None,
                expand_children=1, play_out=1, play_out_selection="best", use_combo=False, candidate_pool_size=100,
                 combo_lvl=1, combo_init_random=5, combo_step=5, combo_play_out=20, use_PG=False, PG_batch_size=50)


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
