import xrayutilities as xu
import numpy as np
import mdts
import itertools
import statistics as st
import csv


L1_thick=range(5,200,5)
L2_thick=range(5,200,5)
np.random.seed(1)

bil_thick=list(itertools.product(L1_thick, L2_thick))
print (len(bil_thick))

old_range_max=1520
old_range_min=0
new_range_max=78
new_range_min=0

with open("materials") as f:
  matList = f.readlines()

def scale(m):
   m_tmp= (m-old_range_min)/(old_range_max-old_range_min)
   new_m= (m_tmp*(new_range_max-new_range_min))+new_range_min
   return int(round(new_m))

### calculate reflectivty
def simul_ref(struct):
    mat1=matList[scale(struct[0])]
    mat2=matList[scale(struct[1])]


    ls = xu.simpack.LayerStack('mat1/mat2',)


    for i in struct[2:]:
        layer1_str="layer1=xu.simpack.Layer(xu.materials."+mat1+",bil_thick[i][0],roughness=3.0)"
        exec(layer1_str)
        exec("ls.append(layer1)")

        layer2_str="layer2=xu.simpack.Layer(xu.materials."+mat2+",bil_thick[i][1],roughness=3.0)"
        exec(layer2_str)
        exec("ls.append(layer2)")


    # reflectivity calculation
    m = xu.simpack.SpecularReflectivityModel(ls, energy=8050.92162)
    ai = np.arange(0.45, 0.55, 0.01)
    Ixrr = m.simulate(ai)


    #ai_out=np.arange(0.40, 0.44, 0.01)
    #Ixrr_out = m.simulate(ai_out)


    return (st.mean(Ixrr))               ###-(0.1*(st.mean(Ixrr_out)))


def calc_thickness(struct):
    thickness=[]
    t_thickness=0
    for j in struct[2:]:
        thickness.append(bil_thick[j][0])
        t_thickness+=bil_thick[j][0]
        thickness.append(bil_thick[j][1])
        t_thickness+=bil_thick[j][1]

    return thickness, t_thickness

def matof(struct):
    return matList[scale(struct[0])], matList[scale(struct[1])]

myTree=mdts.Tree(no_positions=5, atom_types=list(range(1521)), atom_const=None, get_reward=simul_ref, positions_order="direct",
                max_flag=True,expand_children=2, play_out=1, play_out_selection="best", space=None, candidate_pool_size=100,
                 ucb="mean", use_combo=False, combo_play_out=20, combo_init_random=5, combo_step=5, combo_lvl=5)

res=myTree.search_PG(display=True,no_candidates=20000)# apply MCTS search with PG 




### Optimal reward
print (res.optimal_fx)

### List of optimal candidates
print (res.optimal_candidate)

### Number of examined candidates
print (res.checked_candidates_size)

### Maximum depth reached
print (res.max_depth_reached)

### Number of nodes constructed
print (res.no_nodes)

### Average visit per node
print (res.avg_node_visit)

print (calc_thickness(res.optimal_candidate[0]))

print (matof(res.optimal_candidate[0]))

mdts.save_tree(myTree, 'Tree_file_go_mat13')
with open("result_go_mat.csv", mode='a') as f:
    writer = csv.writer(f)
    writer.writerow([str(res.optimal_fx),str(res.optimal_candidate),str(res.checked_candidates_size),\
        str(res.max_depth_reached),str(res.no_nodes),str(res.avg_node_visit),str(calc_thickness(res.optimal_candidate[0]))])
