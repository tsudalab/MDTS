from __future__ import division
from .node import Node
from .result import Result
import ast
import collections
import math
import random
import sys
from .policygradient import PG
import numpy as np
import combo
from goto import with_goto


class Tree:
    def __init__(self, no_positions=None, position_values=None, position_values_const=None, positions_order="direct", max_flag=True, get_reward=None, constraints=None, expand_children=1, play_out=1, play_out_selection="best", use_combo=False, candidate_pool_size=None, combo_lvl=1, combo_init_random=1, combo_step=1,combo_play_out=10, use_PG=False, PG_batch_size=100):

        if (no_positions is None) or (position_values is None):
            sys.exit("no_positions and position_values should not be None")
        else:
            self.no_positions = no_positions

            if all(isinstance(pv,int) for pv in position_values):
                self.position_values = position_values
                self.position_values_lists=[position_values] * self.no_positions
                self.position_values_const = position_values_const
            else:
                if (all(isinstance(pv, list) for pv in position_values)) and (len(position_values)==self.no_positions):
                    self.position_values_lists=position_values
                    flat_pv_list = [item for sublist in position_values for item in sublist]
                    self.position_values=list(set(flat_pv_list))
                    self.position_values_const = None
                else:
                    sys.exit("no_positions and position_values do not match")


        if positions_order == "direct":
            self.positions_order = list(range(self.no_positions))
        elif positions_order == "reverse":
            self.positions_order = list(range(self.no_positions))[::-1]
        elif positions_order == "shuffle":
            self.positions_order = random.sample(
                list(range(self.no_positions)), self.no_positions)
        elif isinstance(positions_order, list):
            self.positions_order = positions_order
        else:
            sys.exit("Please specify positions order as a list")

        self.max_flag = max_flag

        if get_reward==None:
            sys.exit("Reward function can not be None")
        else:
            self.get_reward = get_reward

        self.constraints = constraints

        if expand_children == "all":
            self.expand_children = len(self.position_values)
        elif isinstance(expand_children, int):
            if (expand_children > len(self.position_values)) or (expand_children == 0):
                sys.exit("Please choose appropriate number of children to expand")
            else:
                self.expand_children = expand_children

        if play_out==0:
            sys.exit("play_out can not be 0")
        else:
            self.play_out = play_out

        if play_out_selection == "best":
            self.play_out_selection_mean = False
        elif play_out_selection == "mean":
            self.play_out_selection_mean = True
        else:
            sys.exit("Please set play_out_selection to either mean or best")

        self.use_combo=use_combo
        if self.use_combo:
            self.combo_init_random = combo_init_random
            self.combo_step = combo_step
            self.combo_lvl = combo_lvl
            self.combo_play_out = combo_play_out
            if (candidate_pool_size is None):
                sys.exit("Please set candidate_pool_size for combo search")
            else:
                self.candidate_pool_size = candidate_pool_size

        ### for policy gradient
        self.use_PG = use_PG
        if use_PG:
            state_dim = self.no_positions+1+len(self.position_values)+1  # is the state (the node) witch has 2 values [level, value] in hot encoded manner
            self.PG = PG(state_dim, len(self.position_values))
        self.PG_batch_size=PG_batch_size
        self.PG_batch={}
        self.PG_batch["states"] =[]
        self.PG_batch["actions"] =[]
        self.PG_batch["rewards"] =[]
        self.PG_trained=False

        self.result = Result()
        self.chkd_candidates = collections.OrderedDict()
        self.root = Node(value='R', children_values_idx=list(range(len(self.position_values))), struct=[None] * self.no_positions)
        ### initializing the root node
        position_child = self.positions_order[0]
        invalid_nodes_values_idx = [self.position_values.index(i) for i in
                                    list(set(self.position_values) - set(self.position_values_lists[position_child]))]
        for inv in invalid_nodes_values_idx:
            self.root.children[inv] = None

        self.acc_threshold = 0.1

    def _fill_cand(self,structure):
        cand = structure[:]
        if self.position_values_const is not None:
            for value_id in range(len(self.position_values)):
                if cand.count(self.position_values[value_id]) > self.position_values_const[value_id]:
                    return "out of constraints"

            for value_id in range(len(self.position_values)):
                diff = self.position_values_const[value_id] - cand.count(self.position_values[value_id])
                if diff != 0:
                    avl_pos = [i for i, x in enumerate(cand) if x is None]
                    to_fill_pos = np.random.choice(avl_pos, diff, replace=False)
                    for pos in to_fill_pos:
                        cand[pos] = self.position_values[value_id]
        else:
            avl_pos = [i for i, x in enumerate(cand) if x is None]
            for pos in avl_pos:
                cand[pos] = np.random.choice(self.position_values_lists[pos])

        if self.constraints is not None:
            if self.constraints(cand):
                return cand
            else:
                return None
        else:
            return cand

    def _enumerate_cands(self, struct, size):
        structure = struct[:]
        chosen_candidates = []

        for pout in range(size):
            cand = structure[:]
            outcand = self._fill_cand(cand)
            if outcand== "out of constraints":
                return chosen_candidates
            else:
                if outcand !=None:
                    chosen_candidates.append(outcand[:])

        i=0
        while (len(chosen_candidates)==0) and (i<=1000000):
            for pout in range(size):
                cand = structure[:]
                outcand=self._fill_cand(cand)
                i+=1
                if outcand !=None:
                    chosen_candidates.append(outcand[:])

        return chosen_candidates

    def one_hot_encode(self, space):
        no_values = len(self.position_values)
        new_space = np.empty((space.shape[0], space.shape[1], no_values),dtype=int)
        for val_ind, val in enumerate(self.position_values):
            one_hot = np.zeros(no_values, dtype=int)
            one_hot[val_ind] = 1
            new_space[space == val] = one_hot
        return new_space.reshape(space.shape[0], space.shape[1] * no_values)

    def _simulate(self, struct):
        if self.use_combo:
            my_space = self._enumerate_cands(struct, self.candidate_pool_size)
            return self._simulate_combo(np.array(my_space))
        else:
            return self._enumerate_cands(struct, self.play_out)

    def get_best_e(self,all_struct):
        rewards = []
        best_e=None
        for struct in all_struct:
            if str(struct) not in self.chkd_candidates.keys():
                e = self.get_reward(struct)
                self.chkd_candidates[str(struct)] = e
            else:
                e = self.chkd_candidates[str(struct)]
            rewards.append(e)
        if len (rewards)!=0:
            rewards_F = [j for j in rewards if j is False]
            rewards_v = [x for x in rewards if x is not False]
            if len(rewards_v) != 0:
                if self.play_out_selection_mean:
                    best_e = np.mean(rewards)
                else:
                    if self.max_flag:
                        best_e = max(rewards)
                    else:
                        best_e = min(rewards)
            elif len(rewards_F)!=0:
                best_e=False
        return best_e


    def _simulate_combo(self, my_space):
        chosen_candidates = []
        sub_space = my_space
        one_hot_sub_space = self.one_hot_encode(my_space)

        if sub_space.shape[0] != 0:

            def combo_simulater(action):
                if str(list(
                        sub_space[action[0]])) in self.chkd_candidates.keys():
                    if self.max_flag:
                        return self.chkd_candidates[str(list(sub_space[action[0]]))]
                    else:
                        return -self.chkd_candidates[str(list(sub_space[action[0]]))]
                else:
                    if self.max_flag:
                        return self.get_reward(sub_space[action[0]])
                    else:
                        return -self.get_reward(sub_space[action[0]])

            policy = combo.search.discrete.policy(test_X=one_hot_sub_space)

            if self.combo_play_out <= 1:
                sys.exit("combo_play_out can not be less than 2 when use_combo is True")

            sub_space_scand_cand = []
            sub_space_scand_val = []
            for c in self.chkd_candidates.keys():
                t = np.where(np.all(sub_space == ast.literal_eval(c),axis=1))[0]
                if len(t) != 0:
                    sub_space_scand_cand.append(t[0])
                    if self.max_flag:
                        sub_space_scand_val.append(self.chkd_candidates[c])
                    else:
                        sub_space_scand_val.append(-self.chkd_candidates[c])

            sub_space_pair = list(zip(sub_space_scand_cand, sub_space_scand_val))
            sub_space_pair.sort(key=lambda x: x[1], reverse=True)

            if len(sub_space_pair) >= self.combo_play_out:
                for i in range(self.combo_play_out):
                    policy.write(sub_space_pair[i][0], sub_space_pair[i][1])

                trained = self.combo_play_out
            else:
                for x in sub_space_pair:
                    policy.write(x[0], x[1])
                trained = len(sub_space_pair)
                if len(sub_space_pair) < self.combo_init_random:
                    if sub_space.shape[0] >= self.combo_init_random:
                        policy.random_search(max_num_probes=self.combo_init_random - len(sub_space_pair),simulator=combo_simulater)
                        trained = self.combo_init_random
                    else:
                        policy.random_search(max_num_probes=sub_space.shape[0] - len(sub_space_pair),simulator=combo_simulater)
                        trained = sub_space.shape[0]

            if sub_space.shape[0] >= self.combo_play_out:
                res = policy.bayes_search(max_num_probes=self.combo_play_out - trained, simulator=combo_simulater, score='TS', interval=self.combo_step, num_rand_basis=5000)
            else:
                res = policy.bayes_search(max_num_probes=sub_space.shape[0] - trained, simulator=combo_simulater, score='TS', interval=self.combo_step, num_rand_basis=5000)

            for i in range(len(res.chosed_actions[0:res.total_num_search])):
                action = res.chosed_actions[i]
                if self.max_flag:
                    e = res.fx[i]
                else:
                    e = -res.fx[i]
                if str(list(
                        sub_space[action])) not in self.chkd_candidates.keys():
                    self.chkd_candidates[str(list(sub_space[action]))] = e
                chosen_candidates.append(list(sub_space[action]))

        return chosen_candidates


    def hot_lvl(self,lvl):
        no_lvls=self.no_positions+1
        hot_lvl=[0] * no_lvls
        hot_lvl[lvl]=1
        return hot_lvl

    def hot_val(self,val):
        no_vals=len(self.position_values)+1
        hot_val=[0] * no_vals
        if val=='R':
            hot_val[0]=1
        else:
            val_idx=self.position_values.index(val)+1
            hot_val[val_idx]=1
        return hot_val

    def convert_state(self,state):
        hot_lvl=self.hot_lvl(state[0])
        hot_val=self.hot_val(state[1])
        return hot_lvl +hot_val

    def PG_add2batch(self,node):
        PG_rewards = [node.cal_ucb(node.parent.c, self.max_flag)]
        prob_train = [0] * len(self.position_values)
        prob_train[self.position_values.index(node.value)] = 1
        self.PG_batch["states"].append(self.convert_state(node.parent.get_state()))
        self.PG_batch["actions"].append(prob_train)
        self.PG_batch["rewards"].append(PG_rewards)

    @with_goto
    def search(self, no_candidates=None, display=True):
        prev_len = 0
        prev_current = None
        round_no = 1
        if no_candidates is None:
            sys.exit("Please specify no_candidates")
        else:
            while len(self.chkd_candidates) < no_candidates:
                current = self.root.select_origin(self.max_flag)
                label .curr
                if current.level == self.no_positions:
                    struct = current.struct[:]
                    if str(struct) not in self.chkd_candidates.keys():
                        e = self.get_reward(struct)
                        self.chkd_candidates[str(struct)] = e
                    else:
                        e = self.chkd_candidates[str(struct)]

                    current.bck_prop(e)
                    if self.use_PG:
                        self.PG_add2batch(current)

                else:
                    position = self.positions_order[current.level]
                    if current.level==self.no_positions-1:
                        position_child=None
                    else:
                        position_child=self.positions_order[current.level+1]
                    try_children=None
                    if (self.use_PG) and (self.PG_trained):
                        if current.skipped:
                            try_children_idx, try_probs = self.PG.choose_children(self.convert_state(current.get_state()),
                                self.expand_children, current.get_avl_idx(), self.max_flag)
                            try_children = current.expand_PG(position, position_child, try_children_idx, self.position_values, self.position_values_lists)
                        else:
                            try_children_idx, try_probs = self.PG.choose_children(self.convert_state(current.get_state()),
                                                                                  self.expand_children, current.get_all_idx_ex_none(), self.max_flag)
                            expandable, jump_child = current.try_expand_PG(try_children_idx)
                            if expandable:
                                try_children = current.expand_PG(position,position_child, try_children_idx, self.position_values, self.position_values_lists)
                            else:
                                current.skipped=True
                                current = jump_child
                                goto .curr
                    else:
                        try_children = current.expand_origin(position, position_child, self.expand_children, self.position_values, self.position_values_lists)

                    for try_child in try_children:
                        all_struct = self._simulate(try_child.struct)
                        best_e=self.get_best_e(all_struct)
                        if best_e!=None:
                            try_child.bck_prop(best_e)

                            ### to be revised for max or min
                            if self.use_PG:
                                self.PG_add2batch(try_child)
                                #prob_train[self.position_values.index(try_child.value)]=try_probs[try_children_idx.index(self.position_values.index(try_child.value))]

                        else:
                            current.children[try_child.value] = None
                            all_struct = self._simulate(current.struct)
                            best_e = self.get_best_e(all_struct)
                            if best_e!=None:
                                current.bck_prop(best_e)

                                if self.use_PG:
                                    if current.parent!=None:
                                        self.PG_add2batch(current)


                if (current == prev_current) and (len(self.chkd_candidates) == prev_len):
                    if display:
                        print ("adjusting hyperparameter c")
                    adjust_val = (no_candidates - len(self.chkd_candidates)) / no_candidates
                    if adjust_val < self.acc_threshold:
                        adjust_val = self.acc_threshold
                    current.adjust_c(adjust_val)

                if self.use_PG:
                    if len(self.PG_batch["rewards"]) >= self.PG_batch_size:
                        if display:
                            print ("Training policy network")
                        self.PG.train(self.PG_batch, len(self.PG_batch["rewards"]))
                        self.PG_trained = True
                        self.PG_batch["states"] = []
                        self.PG_batch["actions"] = []
                        self.PG_batch["rewards"] = []

                prev_len = len(self.chkd_candidates)
                prev_current = current
                if display:
                    print("round ", round_no)
                    print("checked candidates = ", len(self.chkd_candidates))
                    if len(self.chkd_candidates)!=0:
                        ex_F=[i for i in iter(self.chkd_candidates.values()) if i is not False]
                        if len(ex_F)!=0:
                            if self.max_flag:
                                print("current best = ",
                                      max(ex_F))
                            else:
                                print("current best = ",
                                      min(ex_F))

                round_no += 1

        self.result.format(no_candidates=no_candidates,
                           chkd_candidates=self.chkd_candidates,
                           max_flag=self.max_flag)
        self.result.no_nodes, visits, self.result.max_depth_reached = self.root.get_info()
        self.result.avg_node_visit = visits / self.result.no_nodes
        return self.result