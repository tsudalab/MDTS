from __future__ import division
import numpy as np
import math
from .policygradient import *


class Node:
    def __init__(self, value, children_values, parent=None, level=0, position=None, struct=None, v=0, w=0.0, children=None, min_range=float(math.pow(10, 10)),max_range=0.0, adjust_val=1):

        self.value = value
        self.parent = parent
        self.level = level
        self.position = position
        self.struct = struct
        self.v = v  # visit time
        self.w = w  # the merit
        self.children = children
        self.min_range = min_range
        self.max_range = max_range
        self.adjust_val = adjust_val
        self.children_values = children_values

    def has_children(self):
        return self.children is not None

    def has_all_children(self):
        if self.has_children():

            return len(self.children) == len(self.children_values)
        else:
            return False

    def select_origin(self, max_flag, ucb_mean):  # original selection strategy (expand all children first)
        if self.has_all_children():
            c = ((math.sqrt(2) / 4) * (self.max_range - self.min_range)) * self.adjust_val
            u_scores = {}
            if max_flag:
                for child in iter(self.children.values()):
                    if child is not None:
                        if ucb_mean:
                            u_scores[child.value] = ((child.w / child.v) + (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
                        else:
                            u_scores[child.value] = child.max_range + (c * (math.sqrt((2 * math.log(self.v)) / child.v)))
                max_idxs = [i for i, x in iter(u_scores.items()) if x == max(iter(u_scores.values()))]
                idx = np.random.choice(max_idxs)
            else:
                for child in iter(self.children.values()):
                    if child is not None:
                        if ucb_mean:
                            u_scores[child.value] = ((child.w / child.v) - (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
                        else:
                            u_scores[child.value] = child.min_range + (c * (math.sqrt((2 * math.log(self.v)) / child.v)))
                min_idxs = [i for i, x in iter(u_scores.items()) if x == min(iter(u_scores.values()))]
                idx = np.random.choice(min_idxs)
            return self.children[idx].select_origin(max_flag, ucb_mean)
        else:
            return self

### should be removed
    # def select_root(self, max_flag, ucb_mean):  # select nodes from the root node
    #     if self.has_all_children():
    #         c = ((math.sqrt(2) / 4) * (self.max_range - self.min_range)) * self.adjust_val
    #         u_scores = {}
    #         if max_flag:
    #             for child in iter(self.children.values()):
    #                 if child is not None:
    #                     if ucb_mean:
    #                         u_scores[child.value] = ((child.w / child.v) + (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
    #                     else:
    #                         u_scores[child.value] = child.max_range + (c * (math.sqrt((2 * math.log(self.v)) / child.v)))
    #             max_idxs = [i for i, x in iter(u_scores.items()) if x == max(iter(u_scores.values()))]
    #             idx = np.random.choice(max_idxs)
    #         else:
    #             for child in iter(self.children.values()):
    #                 if child is not None:
    #                     if ucb_mean:
    #                         u_scores[child.value] = ((child.w / child.v) - (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
    #                     else:
    #                         u_scores[child.value] = child.min_range + (c * (math.sqrt((2 * math.log(self.v)) / child.v)))
    #             min_idxs = [i for i, x in iter(u_scores.items()) if x == min(iter(u_scores.values()))]
    #             idx = np.random.choice(min_idxs)
    #         return self.children[idx]
    #     else:  # return ifself if this node doesn't have children
    #         return self

    def cal_ucb(self, node, ucb_mean):  # calculate UCB values
        c = ((math.sqrt(2) / 4) *
             (node.max_range - node.min_range)) * node.adjust_val
        if node.parent is not None:
            if ucb_mean:
                ucb_score = (node.w / node.v) + (c * (math.sqrt(
                    (2 * math.log(node.parent.v)) / node.v)))
            else:
                ucb_score = node.max_range + (c * (math.sqrt(
                    (2 * math.log(node.parent.v)) / node.v)))
        return ucb_score

    def select(self, max_flag, ucb_mean):
        if self.parent is not None:
            if self.has_children():
                u_scores = {}
                for child in iter(self.children.values()):
                    if child is not None:
                        if ucb_mean:

                            u_scores[child.value] = self.cal_ucb(
                                child, ucb_mean)
                        else:
                            u_scores[child.value] = self.cal_ucb(
                                child, ucb_mean=False)
                if ucb_mean:
                    u_scores[self.value] = self.cal_ucb(self, ucb_mean)
                else:
                    u_scores[self.value] = self.cal_ucb(self, ucb_mean=False)

                if max_flag:
                    max_idxs = [
                        i for i, x in iter(u_scores.items())
                        if x == max(iter(u_scores.values()))
                    ]
                    idx = np.random.choice(max_idxs)
                else:
                    min_idxs = [
                        i for i, x in iter(u_scores.items())
                        if x == min(iter(u_scores.values()))
                    ]
                    idx = np.random.choice(min_idxs)

                if idx == self.value:
                    return self
                else:
                    return self.children[idx].select(max_flag, ucb_mean)
            else:
                return self
        else:
            if self.has_all_children():
                u_scores = {}
                for child in iter(self.children.values()):
                    if child is not None:
                        if ucb_mean:

                            u_scores[child.value] = self.cal_ucb(
                                child, ucb_mean)
                        else:
                            u_scores[child.value] = self.cal_ucb(
                                child, ucb_mean=False)
                if max_flag:
                    max_idxs = [
                        i for i, x in iter(u_scores.items())
                        if x == max(iter(u_scores.values()))
                    ]
                    idx = np.random.choice(max_idxs)
                else:
                    min_idxs = [
                        i for i, x in iter(u_scores.items())
                        if x == min(iter(u_scores.values()))
                    ]
                    idx = np.random.choice(min_idxs)
                return self.children[idx]
            else:
                return self

    def bck_prop(self, e):
        self.v += 1
        self.w += e
        if e > self.max_range:
            self.max_range = e
        if e < self.min_range:
            self.min_range = e
        if self.parent is not None:
            return self.parent.bck_prop(e)

    def adjust_c(self, adjust_value):
        self.adjust_val += adjust_value
        if self.parent is not None:
            return self.parent.adjust_c(adjust_value)

    def choose_expand(self, expand_children, state, random, skip_chosen):
        if self.children is None:
            self.children = {}

        avl_child_values = list(
            set(self.children_values) - set(self.children.keys()))
        print('avl_child_values', avl_child_values)

        if expand_children > len(avl_child_values):
            no_chosen_values = len(avl_child_values)
        else:
            no_chosen_values = expand_children
        if random:  # randomly select node to expand

            prob = []
            chosen_values = np.random.choice(avl_child_values,
                                             no_chosen_values,
                                             replace=False)
        else:
            if skip_chosen:  # allow for selection of selected nodes
                chosen_values, prob = choose_action_skip(
                    avl_child_values, state, no_chosen_values)
            else:  # not allow for selection of selected nodes
                chosen_values, prob = choose_action(
                    avl_child_values, state, no_chosen_values)

        return chosen_values, prob

    def expand_origin(self, position, expand_children):
        if self.children is None:
            self.children = {}
        expanded = []
        avl_child_values = list(set(self.children_values) - set(self.children.keys()))
        if expand_children > len(avl_child_values):
            no_chosen_values = len(avl_child_values)
        else:
            no_chosen_values = expand_children
        chosen_values = np.random.choice(avl_child_values, no_chosen_values, replace=False)
        for child_value in chosen_values:
            child_struct = self.struct[:]
            child_struct[position] = child_value
            self.children[child_value] = Node(value=child_value, children_values=self.children_values, parent=self,
                                              level=self.level + 1, position=position, struct=child_struct)
            expanded.append(self.children[child_value])
        return expanded


    def expand(self, chosen_values, position):
        if self.children is None:
            self.children = {}
        expanded = []
        for child_value in chosen_values:

            child_struct = self.struct[:]
            child_struct[position] = child_value
            self.children[child_value] = Node(
                value=child_value,
                children_values=self.children_values,
                parent=self,
                level=self.level + 1,
                position=position,
                struct=child_struct)
            expanded.append(self.children[child_value])
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
