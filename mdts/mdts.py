from __future__ import division
from node import Node
from result import Result
import collections
import random
import numpy as np
import sys

class Tree:
    def __init__(self, get_reward, positions_order="direct", max_flag=True, expand_children=1,
                 data=None, no_features=None, feature_values=None, values_const=None, play_out=1):

        self.data = data
        if data is None:
            if (no_features is None) or (feature_values is None):
                sys.exit("no_features and feature_values should not be None")
            else:
                self.no_features = no_features
                self.feature_values = feature_values
                self.values_const = values_const
        else:
            self.no_features = data.shape[1]
            self.feature_values = np.unique(data)
        if positions_order == "direct":
            self.positions_order = range(self.no_features)
        elif positions_order == "reverse":
            self.positions_order = range(self.no_features)[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(range(self.no_features), self.no_features)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("please specify positions order")
        self.chkd_candidates = collections.OrderedDict()
        self.max_flag = max_flag
        self.root = Node(value='R', children_values=self.feature_values, struct=[None]*self.no_features)
        self.acc_threshold = 0.1
        self.get_reward = get_reward
        if expand_children == "all":
            self.expand_children = len(self.feature_values)
        elif isinstance(expand_children, int):
            if (expand_children > len(self.feature_values)) or (expand_children == 0):
                sys.exit("please choose appropriate number of children to expand")
            else:
                self.expand_children = expand_children
        self.result = Result()
        self.play_out = play_out

    def _simulate(self, struct):
        if self.data is None:
            return self._simulate_const(struct)
        else:
            return self._simulate_matrix(struct)

    def _simulate_const(self, struct):
        structure = struct[:]
        chosen_candidates = []
        if self.values_const is not None:
            for value_id in range(len(self.feature_values)):
                if structure.count(self.feature_values[value_id]) > self.values_const[value_id]:
                    return chosen_candidates
            for i in range(self.play_out):
                cand = structure[:]
                for value_id in range(len(self.feature_values)):
                    diff = self.values_const[value_id] - cand.count(self.feature_values[value_id])
                    if diff != 0:
                        avl_pos = [i for i, x in enumerate(cand) if x is None]
                        to_fill_pos = np.random.choice(avl_pos, diff, replace=False)
                        for pos in to_fill_pos:
                            cand[pos] = self.feature_values[value_id]
                chosen_candidates.append(cand)
        else:
            for i in range(self.play_out):
                cand = structure[:]
                avl_pos = [i for i, x in enumerate(cand) if x is None]
                for pos in avl_pos:
                    cand[pos] = np.random.choice(self.feature_values)
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

    def search(self, no_candidates, display=True):
        prev_len = 0
        prev_current = None
        round_no = 1
        while len(self.chkd_candidates) < no_candidates:
            current = self.root.select(self.max_flag)
            if current.level == self.no_features:
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
                    all_struct = self._simulate(try_child.struct)
                    if len(all_struct) != 0:
                        rewards = []
                        for struct in all_struct:
                            if str(struct) not in self.chkd_candidates.keys():
                                e = self.get_reward(struct)
                                self.chkd_candidates[str(struct)] = e
                            else:
                                e = self.chkd_candidates[str(struct)]
                            rewards.append(e)
                        if self.max_flag:
                            best_e = max(rewards)
                        else:
                            best_e = min(rewards)
                        try_child.bck_prop(best_e)
                    else:
                        current.children[try_child.value] = None
                        all_struct = self._simulate(current.struct)
                        rewards = []
                        for struct in all_struct:
                            if str(struct) not in self.chkd_candidates.keys():
                                e = self.get_reward(struct)
                                self.chkd_candidates[str(struct)] = e
                            else:
                                e = self.chkd_candidates[str(struct)]
                            rewards.append(e)
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
            else:
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
        if len(self.chkd_candidates) > no_candidates:
            for i in range(len(self.chkd_candidates) - no_candidates):
                self.chkd_candidates.popitem()
        self.result.format(chkd_candidates=self.chkd_candidates, max_flag=self.max_flag)
        visits = 0
        self.result.no_nodes, visits, self.result.max_depth_reached = self.root.get_info()
        self.result.avg_node_visit = visits / self.result.no_nodes
        return self.result
