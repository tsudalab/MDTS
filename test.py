import mdts
import numpy as np

DB_16=np.load('./DB_16.npy').item()


def simulate(val):
    no_replace = 8
    structure = val[:]
    avl_pos = [i for i, x in enumerate(structure) if x == None]
    no_ones = structure.count(1)

    if len(avl_pos) != 0:
        rand_ones_pos = np.random.choice(avl_pos, no_replace - no_ones, replace=False)

        for pos in rand_ones_pos:
            structure[pos] = 1

        for pos in range(len(structure)):
            if structure[pos] == None:
                structure[pos] = 0

    return structure


def get_reward(struct):
    s = ''.join(str(x) for x in struct)
    if (s in DB_16.keys()) == True:
        cond = DB_16[s]
    else:
        print "error"

    return cond


def node_status(val):
    no_replace = 8
    ones = val.count(1)
    zeros=val.count(0)
    if (ones==no_replace) or (zeros==no_replace):
        return 1 # all ones  or zeros have added,rest are zeros
    else:
        return 0 #continue normal exapnsion


x=mdts.Tree(no_features=16, feature_values=[0,1],get_reward=get_reward, positions_order=range(16), no_candidates=1000,
            max_flag=True,simulate=simulate,chk_node_const=node_status)

optimal,samples=x.search()

