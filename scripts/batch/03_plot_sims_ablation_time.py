#!/usr/bin/env python
#
# Created on: 2022-03-31
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_num_sims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

a_texts = [f'NN {v} sims' for v in all_num_sims]

timeing = np.zeros(len(all_num_sims))

for i in range(len(all_num_sims)):
    file1 = f'../../data/nn_ablation_128_5/timeing_minimax2_vs_nn{all_num_sims[i]}.csv'
    df1 = pd.read_csv(file1)
    timeing[i] = df1['avg_time_black']

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(all_num_sims, timeing, 'o-')
ax.set_xlabel('Number of MCTS simulations')
ax.set_ylabel('Time [s]')
# ax.set_label('Average computation time per turn')
fig.savefig('/tmp/sims_ablation_time.png')
# plt.show()
