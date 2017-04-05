import mdts
import numpy as np



### Thermal conductivity for (Si:Ge=1:1) alloy with 16 atoms in the interfacial structure
DB_16=np.load('./DB_16.npy').item()

### the get_reward simulates the structure: takes a python list represents the structure and return the reward
def get_reward(struct):
    s = ''.join(str(x) for x in struct)
    if s in DB_16.keys():
        cond = DB_16[s]
    else:
        print "error"
    return cond


### initialize the tree with the following parameters
### no_features: number of features in each structure. For example, 16 atoms.
### feature_values: possible values for each feature. For example, types of atoms: 0 for Si and 1 for Ge
### values_const: number of each feature value in the . For example, 8 atoms Si and 8 atoms Ge default is None
### get_reward: the simulation function
### positions_order: define the order to assign feature values to the positions in the structure: "direct", "reverse",
    #shuffle# or a list
### max_flag: if True the algorithm searches for maximum reward, else for minimum
### expand_children : number of children to expand at each node. Default is 1
### play_out: number of play outs et each node. Default is 1
### data: numpy ndarray representing the candidates space. Default is None. If specified the "no_features",
    # "feature_values", and "values_const" parameters will be ignored and there values will be taken from the data.
    # This is a slower option, not recommended unless there are complex constraints on the structures


myTree=mdts.Tree(no_features=16, feature_values=[0,1], values_const=[8,8], get_reward=get_reward, positions_order=range(16),
                max_flag=True,expand_children=2, play_out=2, data=None)

### start the search for certain number of candidates and returns an object of type Result contains the result of the search
res=myTree.search(display=False,no_candidates=1000)

### optimal reward
print res.optimal_fx

### list of optimal candidates
print res.optimal_candidate

### list of tuples with the candidates examined and their rewards
print res.checked_candidates_DS

### number of examined candidates
print res.checked_candidates_size

### list of chosen candidates in order
print res.checked_candidates

### list of simulated candidates rewards in order
print res.fx

### List of current best reward
print res.best_fx

### maximum depth reached
print res.max_depth_reached

### number of nodes constructed
print res.no_nodes

### average visit per node
print res.avg_node_visit

### save results
res.save('results.npz')

### load results
new_res=mdts.Result()
new_res.load('results.npz')
print new_res.optimal_fx


### the tree can be saved
mdts.save_tree(myTree,'Tree_file')
del myTree

### load the tree and continue the search
myNewTree = mdts.load_tree('Tree_file')
newTree_res=myNewTree.search(display=False, no_candidates=1200)
print newTree_res.checked_candidates_size
