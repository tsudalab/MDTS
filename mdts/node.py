from __future__ import division
import numpy as np
import math


class Node:
    def __init__(self, value, children_values, parent=None, level=0, position=None, struct=None, v=0, w=0.0,
                 children=None, min_range=float(math.pow(10, 10)), max_range=0.0, adjust_val=1):

        self.value = value
        self.parent = parent
        self.level = level
        self.position = position
        self.struct = struct
        self.v = v
        self.w = w
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

    def select(self, max_flag):
        if self.has_all_children():
            c = ((math.sqrt(2)/4)*(self.max_range-self.min_range))*self.adjust_val
            u_scores = {}
            if max_flag:
                for child in self.children.itervalues():
                    if child is not None:
                        u_scores[child.value] = ((child.w / child.v) + (c * (math.sqrt((2 * math.log(self.v)) /
                                                                                       child.v))))
                max_idxs = [i for i, x in u_scores.iteritems() if x == max(u_scores.itervalues())]
                idx = np.random.choice(max_idxs)
            else:
                for child in self.children.itervalues():
                    if child is not None:
                        u_scores[child.value] = ((child.w / child.v) - (c * (math.sqrt((2 * math.log(self.v)) /
                                                                                       child.v))))
                min_idxs = [i for i, x in u_scores.iteritems() if x == min(u_scores.itervalues())]
                idx = np.random.choice(min_idxs)
            return self.children[idx].select(max_flag)
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

    def expand(self, position, expand_children):
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

    def get_info(self):
        nodes = 1
        visits = self.v
        depth = self.level
        if self.has_children():
            for child in self.children.itervalues():
                if child is not None:
                    x, y, z = child.get_info()
                    nodes +=x
                    visits += y
                    if z > depth:
                        depth = z
        return nodes, visits, depth

