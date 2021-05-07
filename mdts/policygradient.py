import numpy as np
from tensorflow.keras import models, layers, optimizers


class PG:
    def __init__(self, input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = models.Sequential([
            layers.Dense(2048, input_dim=input_dim, activation='relu'),
            layers.Dense(output_dim, activation="softmax")
        ])
        self.model.compile(loss='mean_squared_error', optimizer=optimizers.Adam())


    def choose_children(self, state, size,avl_idx, max_flag):  # choose the nodes via policy gradient
        probs = self.model.predict(np.array(state).reshape(1,self.input_dim))[0]
        prob_zip=zip(range(len(probs)), probs)

        sel_probs_zip=[(i,v) for (i,v) in prob_zip if i in avl_idx]

        if max_flag:
            sorted_sel_probs_zip = sorted(sel_probs_zip, key=lambda v: v[1], reverse=True)
        else:
            sorted_sel_probs_zip = sorted(sel_probs_zip, key=lambda v: v[1], reverse=False)

        sorted_idx = [i[0] for i in sorted_sel_probs_zip]
        sorted_sel_probs = [i[1] for i in sorted_sel_probs_zip]

        if size > len(sorted_sel_probs):
            no_chosen_values = len(sorted_sel_probs)
        else:
            no_chosen_values = size

        return sorted_idx[:no_chosen_values],sorted_sel_probs[:no_chosen_values]


    def train(self, PG_batch, PG_batch_size):
        self.model.fit(np.array(PG_batch["states"]).reshape(PG_batch_size, self.input_dim), np.array(PG_batch["actions"]).reshape(PG_batch_size,self.output_dim), sample_weight=np.array(PG_batch["rewards"]).flatten(), verbose=0)