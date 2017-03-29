from __future__ import division
import numpy as np
import math


class Node:
    def __init__(self, parent=None, level=0, position=None, val=None, v=0, w=0.0, children=None,
                 min_range=float(math.pow(10, 10)), max_range=0.0, adjust_val=1):

        self.parent = parent
        self.level = level
        self.position = position
        self.val = val
        self.v = v
        self.w = w
        self.children = children
        self.min_range = min_range
        self.max_range = max_range
        self.adjust_val = adjust_val

    def has_children(self):
        return self.children is not None

    def select(self, max_flag):
        if self.has_children():
            c = ((math.sqrt(2)/4)*(self.max_range-self.min_range))*self.adjust_val
            u_scores = []
            if max_flag:
                for child in self.children:
                    u_scores.append((child.w / child.v) + (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
                max_idxs = [i for i, x in enumerate(u_scores) if x == max(u_scores)]
                idx = np.random.choice(max_idxs, 1)[0]
            else:
                for child in self.children:
                    u_scores.append((child.w / child.v) - (c * (math.sqrt((2 * math.log(self.v)) / child.v))))
                min_idxs = [i for i, x in enumerate(u_scores) if x == min(u_scores)]
                idx = np.random.choice(min_idxs, 1)[0]

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

    def expand(self, values, position):
        self.children = []
        for i in range(len(values)):
            child_val = self.val[:]
            child_val[position] = values[i]
            self.children.append(Node(parent=self, level=self.level + 1, position=position, val=child_val))
        return self.children
