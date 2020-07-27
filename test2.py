import xrayutilities as xu
import numpy as np
import mdts
import itertools
import statistics as st
import csv

L1_thick = range(5, 200, 5)
L2_thick = range(5, 200, 5)

bil_thick = list(itertools.product(L1_thick, L2_thick))


### calculate reflectivty
def simul_ref(struct):
    ls = xu.simpack.LayerStack('W/C', )

    for i in struct:
        layer1 = xu.simpack.Layer(xu.materials.C,
                                  bil_thick[i][0],
                                  roughness=3.0)
        ls.append(layer1)
        layer2 = xu.simpack.Layer(xu.materials.W,
                                  bil_thick[i][1],
                                  roughness=3.0)
        ls.append(layer2)

    # reflectivity calculation
    m = xu.simpack.SpecularReflectivityModel(ls, energy=8050.92162)
    ai = np.arange(0.45, 0.55, 0.01)
    Ixrr = m.simulate(ai)

    return (st.mean(Ixrr))


def calc_thickness(struct):
    thickness = []
    t_thickness = 0
    for j in struct:
        thickness.append(bil_thick[j][0])
        t_thickness += bil_thick[j][0]
        thickness.append(bil_thick[j][1])
        t_thickness += bil_thick[j][1]

    return thickness, t_thickness


myTree = mdts.Tree(no_positions=20,
                   atom_types=list(range(1521)),
                   atom_const=None,
                   get_reward=simul_ref,
                   positions_order="direct",
                   max_flag=True,
                   expand_children=2,
                   play_out=1,
                   play_out_selection="best",
                   space=None,
                   candidate_pool_size=100,
                   ucb="mean",
                   use_combo=False,
                   combo_play_out=20,
                   combo_init_random=5,
                   combo_step=5,
                   combo_lvl=5)

res = myTree.search_PG(display=True,
                       no_candidates=100)  # apply MCTS search with PG

#res=myTree.search(display=True,no_candidates=20000)# apply MCTS search without PG


### Optimal reward
print(res.optimal_fx)

### List of optimal candidates
print(res.optimal_candidate)

### Number of examined candidates
print(res.checked_candidates_size)

### Maximum depth reached
print(res.max_depth_reached)

### Number of nodes constructed
print(res.no_nodes)

### Average visit per node
print(res.avg_node_visit)

print(calc_thickness(res.optimal_candidate[0]))

mdts.save_tree(myTree, 'Tree_file')

