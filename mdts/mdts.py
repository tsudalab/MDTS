from __future__ import division
from node import Node
from result import Result
import collections
import random
import numpy as np
import sys
import combo
import math
import ast


class Tree:
    def __init__(self, get_reward, positions_order="reverse", max_flag=True, expand_children=1,
                 data=None, no_positions=None, atom_types=None, atom_const=None, play_out=1, play_out_selection="best",
                 ucb="mean", use_combo=False, combo_init_random=1, combo_step=1, combo_lvl=1, combo_play_out=10):

        if data is None:
            self.data=None
            if (no_positions is None) or (atom_types is None):
                sys.exit("no_positions and atom_types should not be None")
            else:
                self.no_positions = no_positions
                self.atom_types = atom_types
                self.atom_const = atom_const
        else:
            self.data = data.copy()
            self.no_positions = data.shape[1]
            self.atom_types = np.unique(data)
        if positions_order == "direct":
            self.positions_order = range(self.no_positions)
        elif positions_order == "reverse":
            self.positions_order = range(self.no_positions)[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(range(self.no_positions), self.no_positions)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("Please specify positions order")
        self.chkd_candidates = collections.OrderedDict()
        self.max_flag = max_flag
        self.root = Node(value='R', children_values=self.atom_types, struct=[None]*self.no_positions)
        self.acc_threshold = 0.1
        self.get_reward = get_reward
        if expand_children == "all":
            self.expand_children = len(self.atom_types)
        elif isinstance(expand_children, int):
            if (expand_children > len(self.atom_types)) or (expand_children == 0):
                sys.exit("Please choose appropriate number of children to expand")
            else:
                self.expand_children = expand_children
        self.result = Result()
        self.play_out = play_out
        if play_out_selection == "best":
            self.play_out_selection_mean = False
        elif play_out_selection =="mean":
            self.play_out_selection_mean = True
        else:
            sys.exit("Please set play_out_selection to either mean or best")
        self.use_combo = use_combo
        self.combo_init_random = combo_init_random
        self.combo_step = combo_step
        self.combo_lvl = combo_lvl
        self.combo_play_out=combo_play_out
        if use_combo is True and data is None:
            sys.exit("Please set data to be able to use combo")
        if ucb == "best":
            self.ucb_mean = False
        elif ucb =="mean":
            self.ucb_mean = True
        else:
            sys.exit("Please set ucb to either mean or best")

    def _simulate(self, struct, lvl):
        if self.data is None:
            return self._simulate_const(struct)
        elif (self.use_combo) and (lvl >= self.combo_lvl):
            return self._simulate_combo(struct)
        else:
            return self._simulate_matrix(struct)

    def _simulate_const(self, struct):
        structure = struct[:]
        chosen_candidates = []
        if self.atom_const is not None:
            for value_id in range(len(self.atom_types)):
                if structure.count(self.atom_types[value_id]) > self.atom_const[value_id]:
                    return chosen_candidates
            for pout in range(self.play_out):
                cand = structure[:]
                for value_id in range(len(self.atom_types)):
                    diff = self.atom_const[value_id] - cand.count(self.atom_types[value_id])
                    if diff != 0:
                        avl_pos = [i for i, x in enumerate(cand) if x is None]
                        to_fill_pos = np.random.choice(avl_pos, diff, replace=False)
                        for pos in to_fill_pos:
                            cand[pos] = self.atom_types[value_id]
                chosen_candidates.append(cand)
        else:
            for pout in range(self.play_out):
                cand = structure[:]
                avl_pos = [i for i, x in enumerate(cand) if x is None]
                for pos in avl_pos:
                    cand[pos] = np.random.choice(self.atom_types)
                chosen_candidates.append(cand)
        return chosen_candidates

    def _simulate_matrix(self, struct):
        structure = struct[:]
        chosen_candidates = []
        filled_pos = [i for i, x in enumerate(structure) if x is not None]
        filled_values = [x for i, x in enumerate(structure) if x is not None]
        sub_data = self.data[:, filled_pos]
        avl_candidates_idx = np.where(np.all(sub_data == filled_values, axis=1))[0]
        if len(avl_candidates_idx) != 0:
            if self.play_out <= len(avl_candidates_idx):
                chosen_idxs = np.random.choice(avl_candidates_idx, self.play_out)
            else:
                chosen_idxs = np.random.choice(avl_candidates_idx, len(avl_candidates_idx))
            for idx in chosen_idxs:
                chosen_candidates.append(list(self.data[idx]))
        return chosen_candidates


    def _simulate_combo(self, struct):
        structure = struct[:]
        chosen_candidates = []
        filled_pos = [i for i, x in enumerate(structure) if x is not None]
        filled_values = [x for i, x in enumerate(structure) if x is not None]
        sub_data = self.data[:, filled_pos]
        avl_candidates_idx = np.where(np.all(sub_data == filled_values, axis=1))[0]
        sub_space=self.data[avl_candidates_idx]
        if sub_space.shape[0] !=0:
            def combo_simulater(action):
                if str(list(sub_space[action[0]])) in self.chkd_candidates.keys():
                    if self.max_flag:
                        return self.chkd_candidates[str(list(sub_space[action[0]]))]
                    else:
                        return -self.chkd_candidates[str(list(sub_space[action[0]]))]
                else:
                    if self.max_flag:
                        return self.get_reward(sub_space[action[0]])
                    else:
                        return -self.get_reward(sub_space[action[0]])

            policy = combo.search.discrete.policy(test_X=sub_space)
            if self.combo_play_out <= 1:
                sys.exit("combo_play_out can not be less than 2 when use_combo is True")

            sub_space_scand_cand=[]
            sub_space_scand_val=[]
            for c in self.chkd_candidates.keys():
                t=np.where(np.all(sub_space == ast.literal_eval(c), axis=1))[0]
                if len(t) !=0:
                    sub_space_scand_cand.append(t[0])
                    if self.max_flag:
                        sub_space_scand_val.append(self.chkd_candidates[c])
                    else:
                        sub_space_scand_val.append(-self.chkd_candidates[c])

            sub_space_pair=zip(sub_space_scand_cand,sub_space_scand_val)
            sub_space_pair.sort(key=lambda x: x[1],reverse=True)

            if len(sub_space_pair) >= self.combo_play_out:
                for i in range(self.combo_play_out):
                    policy.write(sub_space_pair[i][0],sub_space_pair[i][1])

                trained=self.combo_play_out
            else:
                for x in sub_space_pair:
                    policy.write(x[0],x[1])
                trained=len(sub_space_pair)
                if len(sub_space_pair) < self.combo_init_random:
                    if sub_space.shape[0] >= self.combo_init_random:
                        policy.random_search(max_num_probes=self.combo_init_random-len(sub_space_pair),
                                             simulator=combo_simulater)
                        trained=self.combo_init_random
                    else:
                        policy.random_search(max_num_probes=sub_space.shape[0] - len(sub_space_pair),
                                             simulator=combo_simulater)
                        trained=sub_space.shape[0]

            if sub_space.shape[0] >= self.combo_play_out:
                res = policy.bayes_search(max_num_probes=self.combo_play_out-trained, simulator=combo_simulater
                                      , score='TS', interval=self.combo_step, num_rand_basis=5000)
            else:
                res = policy.bayes_search(max_num_probes=sub_space.shape[0] - trained, simulator=combo_simulater
                                          , score='TS', interval=self.combo_step, num_rand_basis=5000)

            for i in range(len(res.chosed_actions[0:res.total_num_search])):
                action=res.chosed_actions[i]
                if self.max_flag:
                    e=res.fx[i]
                else:
                    e=-res.fx[i]
                if str(list(sub_space[action])) not in self.chkd_candidates.keys():
                    self.chkd_candidates[str(list(sub_space[action]))] = e
                    #action_origin_idx = np.where(np.all(self.data== sub_space[action], axis=1))[0]
                    #self.data = np.delete(self.data,action_origin_idx[0],axis=0)
                chosen_candidates.append(list(sub_space[action]))

        return chosen_candidates


    def search(self, no_candidates=None, display=True):
        prev_len = 0
        prev_current = None
        round_no = 1
        if no_candidates is None :
            sys.exit("Please specify no_candidates")
        else:
            while len(self.chkd_candidates) < no_candidates:
                current = self.root.select(self.max_flag, self.ucb_mean)
                if current.level == self.no_positions:
                    struct = current.struct[:]
                    if str(struct) not in self.chkd_candidates.keys():
                        e = self.get_reward(struct)
                        self.chkd_candidates[str(struct)] = e
                    else:
                        e = self.chkd_candidates[str(struct)]
                    current.bck_prop(e)
                else:
                    position = self.positions_order[current.level]
                    try_children = current.expand(position, self.expand_children)
                    for try_child in try_children:
                        all_struct = self._simulate(try_child.struct,try_child.level)
                        if len(all_struct) != 0:
                            rewards = []
                            for struct in all_struct:
                                if str(struct) not in self.chkd_candidates.keys():
                                    e = self.get_reward(struct)
                                    self.chkd_candidates[str(struct)] = e
                                else:
                                    e = self.chkd_candidates[str(struct)]
                                rewards.append(e)
                            if self.play_out_selection_mean:
                                best_e = np.mean(rewards)
                            else:
                                if self.max_flag:
                                    best_e = max(rewards)
                                else:
                                    best_e = min(rewards)
                            try_child.bck_prop(best_e)
                        else:
                            current.children[try_child.value] = None
                            all_struct = self._simulate(current.struct,current.level)
                            rewards = []
                            for struct in all_struct:
                                if str(struct) not in self.chkd_candidates.keys():
                                    e = self.get_reward(struct)
                                    self.chkd_candidates[str(struct)] = e
                                else:
                                    e = self.chkd_candidates[str(struct)]
                                rewards.append(e)
                            if self.play_out_selection_mean:
                                best_e = np.mean(rewards)
                            else:
                                if self.max_flag:
                                    best_e = max(rewards)
                                else:
                                    best_e = min(rewards)
                            current.bck_prop(best_e)
                if (current == prev_current) and (len(self.chkd_candidates) == prev_len):
                    adjust_val = (no_candidates-len(self.chkd_candidates))/no_candidates
                    if adjust_val < self.acc_threshold:
                        adjust_val = self.acc_threshold
                    current.adjust_c(adjust_val)
                prev_len = len(self.chkd_candidates)
                prev_current = current
                if display:
                    print "round ", round_no
                    print "checked candidates = ", len(self.chkd_candidates)
                    if self.max_flag:
                        print "current best = ", max(self.chkd_candidates.itervalues())
                    else:
                        print "current best = ", min(self.chkd_candidates.itervalues())
                round_no += 1
        self.result.format(no_candidates=no_candidates, chkd_candidates=self.chkd_candidates, max_flag=self.max_flag)
        self.result.no_nodes, visits, self.result.max_depth_reached = self.root.get_info()
        self.result.avg_node_visit = visits / self.result.no_nodes
        return self.result