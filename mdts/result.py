from __future__ import division
import numpy as np
import ast


class Result:
    def __init__(self):
        self.checked_candidates_DS = None
        self.checked_candidates_size = None
        self.checked_candidates = None
        self.fx = None
        self.optimal_fx = None
        self.optimal_candidate = None
        self.best_fx = None
        self.max_depth_reached = 0
        self.no_nodes = 0
        self.avg_node_visit = 0.0

    def format(self, no_candidates, chkd_candidates, max_flag):
        self.checked_candidates_DS = [(ast.literal_eval(x), v) for (x, v) in chkd_candidates.items()]
        if len(self.checked_candidates_DS) > no_candidates:
            self.checked_candidates_DS = self.checked_candidates_DS[:no_candidates]
        self.checked_candidates_size = len(self.checked_candidates_DS)
        self.checked_candidates, self.fx = map(list, zip(*self.checked_candidates_DS))
        self.best_fx = []
        if max_flag:
            self.optimal_fx = max(self.fx)
            self.optimal_candidate = [k for (k, v) in self.checked_candidates_DS if v == self.optimal_fx]
            self.best_fx.append(self.fx[0])
            for x in self.fx[1:]:
                if x > self.best_fx[-1]:
                    self.best_fx.append(x)
                else:
                    self.best_fx.append(self.best_fx[-1])
        else:
            self.optimal_fx = min(self.fx)
            self.optimal_candidate = [k for (k, v) in self.checked_candidates_DS if v == self.optimal_fx]
            self.best_fx.append(self.fx[0])
            for x in self.fx[1:]:
                if x < self.best_fx[-1]:
                    self.best_fx.append(x)
                else:
                    self.best_fx.append(self.best_fx[-1])

    def save(self, filename):
        wrap = {}
        wrap['checked_candidates_DS'] = self.checked_candidates_DS
        wrap['checked_candidates_size'] = self.checked_candidates_size
        wrap['checked_candidates'] = self.checked_candidates
        wrap['fx'] = self.fx
        wrap['optimal_fx'] = self.optimal_fx
        wrap['optimal_candidate'] = self.optimal_candidate
        wrap['best_fx'] = self.best_fx
        wrap['max_depth_reached'] = self.max_depth_reached
        wrap['no_nodes'] = self.no_nodes
        wrap['avg_node_visit'] = self.avg_node_visit
        np.savez(filename, **wrap)

    def load(self, filename):
        wrap = np.load(filename)
        self.checked_candidates_DS = wrap['checked_candidates_DS']
        self.checked_candidates_size = wrap['checked_candidates_size']
        self.checked_candidates = wrap['checked_candidates']
        self.fx = wrap['fx']
        self.optimal_fx = wrap['optimal_fx']
        self.optimal_candidate = wrap['optimal_candidate']
        self.best_fx = wrap['best_fx']
        self.max_depth_reached = wrap['max_depth_reached']
        self.no_nodes = wrap['no_nodes']
        self.avg_node_visit = wrap['avg_node_visit']
