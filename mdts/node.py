from __future__ import division
import numpy as np
import math


class Node:
    def __init__(self, value, children_values_idx, parent=None, level=0, position=None, struct=None, v=0, w=0.0, children=None, min_range=float(math.pow(100, 100)),max_range=0.0, adjust_val=1):

        self.value = value
        self.parent = parent
        self.level = level
        self.position = position
        self.struct = struct
        self.v = v  # no. visits
        self.w = w  # the merit
        if children is None:
            self.children = {}
        self.min_range = min_range
        self.max_range = max_range
        self.adjust_val = adjust_val
        self.c=((math.sqrt(2) / 4) * (self.max_range - self.min_range)) * self.adjust_val
        self.children_values_idx = children_values_idx
        self.skipped=False

    def has_children(self):
        return len(self.children)!=0
        # haschild=False
        # if self.children!=None:
        #     for i in self.children:
        #         if self.children[i]!=None:
        #             haschild=True
        #             break
        # return haschild

    def has_all_children(self):
        if self.has_children():
            return len(self.children) == len(self.children_values_idx)
        else:
            return False
        # hasallchildren = False
        # if self.children !=None:
        #     if len(self.children) == len(self.children_values):
        #         hasallchildren=True
        #         for i in self.children:
        #             if self.children[i]==None:
        #                 hasallchildren=False
        #                 break
        # return hasallchildren

    def cal_ucb(self, c, max_flag):  # calculate UCB value
        ucb_score=0.0
        if self.parent is not None:
            if max_flag:
                ucb_score = ((self.w / self.v) + (c * (math.sqrt((2 * math.log(self.parent.v)) / self.v))))
            else:
                ucb_score = ((self.w / self.v) - (c * (math.sqrt((2 * math.log(self.parent.v)) / self.v))))
        return ucb_score

    def select_origin(self, max_flag):  # original selection strategy (expand all children first)
        if self.has_all_children():
            u_scores = {}
            for child_val_idx in iter(self.children.keys()):
                if self.children[child_val_idx] is not None:
                    u_scores[child_val_idx]=self.children[child_val_idx].cal_ucb(self.c,max_flag)
            if max_flag:
                max_val_idx = [i for i, x in iter(u_scores.items()) if x == max(iter(u_scores.values()))]
                idx = np.random.choice(max_val_idx)
            else:
                min_val_idx = [i for i, x in iter(u_scores.items()) if x == min(iter(u_scores.values()))]
                idx = np.random.choice(min_val_idx)
            return self.children[idx].select_origin(max_flag)
        else:
            return self

    def bck_prop(self, e):
        self.v += 1
        if e is not False:
            self.w += e
            if e > self.max_range:
                self.max_range = e
            if e < self.min_range:
                self.min_range = e
        self.c=((math.sqrt(2) / 4) * (self.max_range - self.min_range)) * self.adjust_val
        if self.parent is not None:
            return self.parent.bck_prop(e)

    def adjust_c(self, adjust_value):
        self.adjust_val += adjust_value
        self.c=((math.sqrt(2) / 4) * (self.max_range - self.min_range)) * self.adjust_val
        if self.parent is not None:
            return self.parent.adjust_c(adjust_value)

    def expand_origin(self, position, position_child, expand_children, position_values, position_values_lists):
        expanded = []
        # active_keys=[i for i in self.children if self.children[i]!=None]
        # avl_child_values = list(set(self.children_values) - set(active_keys))
        avl_child_values_idx = list(set(self.children_values_idx) - set(self.children.keys()))
        if expand_children > len(avl_child_values_idx):
            no_chosen_values_idx = len(avl_child_values_idx)
        else:
            no_chosen_values_idx = expand_children

        chosen_values_idx = np.random.choice(avl_child_values_idx, no_chosen_values_idx, replace=False)

        for child_value_idx in chosen_values_idx:
            child_struct = self.struct[:]
            child_value=position_values_lists[position][child_value_idx]
            child_struct[position] = child_value
            self.children[child_value_idx] = Node(value=child_value, children_values_idx=self.children_values_idx, parent=self,
                                                  level=self.level + 1, position=position, struct=child_struct)

            if position_child is not None:
                invalid_nodes_values_idx=[position_values.index(i) for i in list(set(position_values) - set(position_values_lists[position_child]))]
                for inv in invalid_nodes_values_idx:
                    self.children[child_value_idx].children[inv]=None

            expanded.append(self.children[child_value_idx])

        self.skipped=False

        return expanded

    def get_info(self):
        nodes = 1
        visits = self.v
        depth = self.level
        if self.has_children():
            for child in iter(self.children.values()):
                if child is not None:
                    x, y, z = child.get_info()
                    nodes += x
                    visits += y
                    if z > depth:
                        depth = z
        return nodes, visits, depth

### for policy gradient
    def get_state(self):  # get the state of the node
        state = [self.level, self.value]
        return state

    def get_avl_idx(self):
        return list(set(self.children_values_idx) - set(self.children.keys()))

    def get_all_idx_ex_none(self):
        nons=[i for i in self.children if self.children[i] is None]
        all_idx_ex_none=[j for j in self.children_values_idx if j not in nons]
        return all_idx_ex_none

    def try_expand_PG(self,chosen_values_idx):
        expand = True
        jump_child = None
        if not self.skipped:
            if len(self.children)!=0:
                for child_value_idx in chosen_values_idx:
                    if (child_value_idx in self.children) and (self.children[child_value_idx]!=None):
                        expand=False
                        jump_child=self.children[child_value_idx]
                        break
        return expand, jump_child

    def expand_PG(self, position, position_child, chosen_values_idx, position_values, position_values_lists):
        expanded = []
        for child_value_idx in chosen_values_idx:
            child_struct = self.struct[:]
            child_value=position_values_lists[position][child_value_idx]
            child_struct[position] = child_value
            self.children[child_value_idx] = Node(value=child_value, children_values_idx=self.children_values_idx,
                                                  parent=self,
                                                  level=self.level + 1, position=position, struct=child_struct)
            if position_child is not None:
                invalid_nodes_values_idx = [position_values.index(i) for i in
                                            list(set(position_values) - set(position_values_lists[position_child]))]
                for inv in invalid_nodes_values_idx:
                    self.children[child_value_idx].children[inv] = None

            expanded.append(self.children[child_value_idx])

        self.skipped=False

        return expanded
