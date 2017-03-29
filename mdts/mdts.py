from __future__ import division
from node import Node
import collections
import random
import numpy as np
import sys


class Tree:
    def __init__(self, no_features, feature_values, get_reward, positions_order="direct", no_candidates=100,
                 max_flag=True, simulate=None, chk_node_const=None):
        self.no_features = no_features
        self.feature_values = feature_values
        if positions_order == "direct":
            self.positions_order = range(no_features)
        elif positions_order == "reverse":
            self.positions_order = range(no_features)[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(range(no_features), no_features)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("please specify positions order")
        self.no_candidates = no_candidates
        self.chkd_candidates = collections.OrderedDict()
        self.max_flag = max_flag
        self.root = Node(val=[None]*no_features)
        self.acc_threshold = 0.1
        self.get_reward = get_reward
        self.simulate = simulate
        self.chk_node_const = chk_node_const

    def _simulate(self, val):
        structure = val[:]
        avl_pos = [i for i, x in enumerate(structure) if x is None]
        if len(avl_pos) != 0:
            for pos in avl_pos:
                structure[pos] = np.random.choice(self.feature_values, 1)
        return structure

    def search(self, display=True):
        prev_len = 0
        prev_current = None
        step = 1
        while len(self.chkd_candidates) < self.no_candidates:
            current = self.root.select(self.max_flag)
            if current.level == self.no_features:
                if self.simulate is None:
                    struct = self._simulate(current.val)
                else:
                    struct = self.simulate(current.val)
                s = ''.join(str(x) for x in struct)
                if s not in self.chkd_candidates.keys():
                    e = self.get_reward(struct)
                    self.chkd_candidates[s] = e
                else:
                    e = self.chkd_candidates[s]
                current.bck_prop(e)
            else:
                if (self.chk_node_const is not None) and (self.chk_node_const(current.val) != 0):
                    if self.simulate is None:
                        struct = self._simulate(current.val)
                    else:
                        struct = self.simulate(current.val)
                    s = ''.join(str(x) for x in struct)
                    if s not in self.chkd_candidates.keys():
                        e = self.get_reward(struct)
                        self.chkd_candidates[s] = e
                    else:
                        e = self.chkd_candidates[s]
                    current.bck_prop(e)
                else:
                    position = self.positions_order[current.level]
                    children = current.expand(self.feature_values, position)
                    for child in children:
                        if self.simulate is None:
                            struct = self._simulate(child.val)
                        else:
                            struct = self.simulate(child.val)
                        s = ''.join(str(x) for x in struct)
                        if s not in self.chkd_candidates.keys():
                            e = self.get_reward(struct)
                            self.chkd_candidates[s] = e
                        else:
                            e = self.chkd_candidates[s]
                        child.bck_prop(e)
            if (current == prev_current) and (len(self.chkd_candidates) == prev_len):
                adjust_val = (self.no_candidates-len(self.chkd_candidates))/self.no_candidates
                if adjust_val < self.acc_threshold:
                    adjust_val = self.acc_threshold
                current.adjust_c(adjust_val)
            else:
                prev_len = len(self.chkd_candidates)
                prev_current = current
            if display:
                print "step=", step
                print "chkd_candidates size=", len(self.chkd_candidates)
                if self.max_flag:
                    print "current best=", max(self.chkd_candidates.itervalues())
                else:
                    print "current best=", min(self.chkd_candidates.itervalues())
            step += 1
        if len(self.chkd_candidates) > self.no_candidates:
            for i in range(len(self.chkd_candidates)-self.no_candidates):
                self.chkd_candidates.popitem()
        if self.max_flag:
            optimal = max(self.chkd_candidates.itervalues())
        else:
            optimal = min(self.chkd_candidates.itervalues())
        return optimal, self.chkd_candidates
