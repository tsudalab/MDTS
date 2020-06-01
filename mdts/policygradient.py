import matplotlib.pyplot as plt
#import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers
import heapq
"""
1. max_num: When recording the output of the policy gradient of each node (the probability of selecting 1521 nodes for expansion), instead of recording only the node corresponding to the maximum probability among them, it records nodes corresponding to the largest max_num probabilities.
2. chosen_num: Monte Carlo tree search has four steps: selection, expansion, simulation and backpropagation, each node will record the maximum max_num nodes after each iteration. After n iterations, n * max_num nodes will be recorded. When certain conditions are met, choose chosen_num nodes among them as the next level, in other words, let the tree grow in this direction without considering other possible child nodes of its parent node.
3. chosen_limit: The conditions mentioned above is: there are chosen_num nodes in the recorded n * max_num nodes which all appear more than chosen_limit times.
4. chose_scl: Parameter named expand_children in original algorithm means the number of nodes that are expanded and simulated in each iteration. The meaning of this parameter is to select chose_scl * expand_children nodes of highest probability among the output of policy gradient, and then select expand_children nodes among these nodes for expansion and simulation. 
"""

"Define network structure, this part is completely tasks dependent, the network structure requires adjustment for different tasks"
STATE_DIM, ACTION_DIM = 8, 1521
model = models.Sequential([
    layers.Dense(2048, input_dim=STATE_DIM, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(ACTION_DIM, activation="softmax")
])
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())


def choose_action_skip(
        avl_child_values, state,
        size):  # choose the nodes via policy gradient(skip chosen)
    chose_scl = 1
    max_num = chose_scl * size
    prob_avl = []
    prob_row = list(model.predict(np.array([state]))[0])
    print("prob_row", max(prob_row))
    for i in prob_row:
        if prob_row.index(i) in avl_child_values:
            prob_avl.append(i)
        else:
            prob_avl.append(0)
    if len(avl_child_values) > max_num:
        max_indexs = [
            prob_avl.index(i) for i in heapq.nlargest(max_num, prob_avl)
        ]
    else:
        max_indexs = avl_child_values[:]
    prob_need = [(prob_row[i]) for i in max_indexs]
    probility = [(1 / sum(prob_need)) * i for i in prob_need]
    print("avl_child_values:", len(avl_child_values))
    if max_indexs:
        choice = np.random.choice(max_indexs, size, p=probility, replace=False)
    else:
        choice = np.random.choice(len(prob_row),
                                  size,
                                  p=model.predict(np.array([state]))[0],
                                  replace=False)

    print("choice", choice)
    return choice, prob_row  #output choice and probability


def choose_action(
        avl_child_values, state,
        size):  # choose the nodes via policy gradient(not skip chosen)
    chose_scl = 1
    max_num = chose_scl * size
    prob_row = list(model.predict(np.array([state]))[0])
    if len(avl_child_values) > max_num:
        max_indexs = [
            prob_row.index(i) for i in heapq.nlargest(max_num, prob_row)
        ]
    else:
        max_indexs = avl_child_values[:]

    prob_need = [(prob_row[i]) for i in max_indexs]
    probility = [(1 / sum(prob_need)) * i for i in prob_need]

    if max_indexs:
        choice = np.random.choice(max_indexs, size, p=probility, replace=False)
    else:
        choice = np.random.choice(len(prob_row),
                                  size,
                                  p=model.predict(np.array([state]))[0],
                                  replace=False)
    print(choice)
    return choice, prob_row


def discount_rewards(rewards, gamma=0.95):
    prior = 0
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out / np.std(out - np.mean(out))


def train(records):

    s_batch = np.array([record[1] for record in records])
    a_batch = np.array(
        [[1 if i in record[2] else 0 for i in range(ACTION_DIM)]
         for record in records])

    prob_batch = model.predict(s_batch) * a_batch

    r_batch = 500 * np.array([record[0] for record in records])**6

    print("r_batch=", r_batch)

    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)
