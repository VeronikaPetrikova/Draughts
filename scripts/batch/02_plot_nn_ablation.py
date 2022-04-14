#!/usr/bin/env python
#
# Created on: 2022-03-15
#
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

# all_params+=("-mcts_num_sims 128 -nn_num_channels 64 -nn_lin_size 256")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 128 -nn_lin_size 512")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 32 -nn_lin_size 128")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 16 -nn_lin_size 64")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 8 -nn_lin_size 32")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 256 -nn_lin_size 1024")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 512 -nn_lin_size 2048")
# all_params+=("-mcts_num_sims 128 -nn_num_channels 1024 -nn_lin_size 4084")
folders = [
    'nn_ablation_128_4',
    'nn_ablation_128_3',
    'nn_ablation_128_2',
    'nn_ablation_128_0',
    'nn_ablation_128_1',
    'nn_ablation_128_5',
    'nn_ablation_128_6',
    'nn_ablation_128_7',
]

conv_size = [8, 16, 32, 64, 128, 256, 512, 1024]
lin_size = [32, 64, 128, 256, 512, 1024, 2048, 4084]
nn_labels = [f'{a} x {b}' for a, b in zip(conv_size, lin_size)]

fig: plt.Figure
fig, axes = plt.subplots(2, 8, squeeze=False, sharey=True, sharex=True, figsize=(6.4 * 3, 4.8))
fig.subplots_adjust(left=0.07, bottom=0.15, right=0.99, top=0.9, wspace=0.1, hspace=0.1)

sum_white_black = True

timing = np.zeros(len(nn_labels))
for minimax in [2, 4]:
    for i in range(len(folders)):
        ax: plt.Axes = axes[0 if minimax == 2 else 1, i]
        file1 = f'../../data/{folders[i]}/progress_minimax{minimax}_vs_nn.csv'
        file2 = f'../../data/{folders[i]}/progress_nn_vs_minimax{minimax}.csv'
        if not Path(file1).exists() or not Path(file2).exists():
            continue
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        labels = df1['White player'].astype(str) + ' vs ' + df1['Black player'].astype(str) + ' ' + df1['note'].astype(
            str)
        if minimax == 2:
            timing[i] = (df1['avg_time_black'].mean() + df2['avg_time_white'].mean()) / 2

        toplot = ['num_white_wins', 'num_draws', 'num_black_wins']
        toplot_r = reversed(toplot)

        my_cmap = plt.cm.get_cmap('copper')
        colors = ['lightcoral', 'silver', 'tab:green']
        bottom = None

        for k, kr, color in zip(toplot, toplot_r, colors):
            if bottom is None:
                bottom = 0 * df1[k]
            d = df1[k] + df2[kr] if sum_white_black else df1[k]
            ax.bar(range(len(df1[k])), d, bottom=bottom, color=color)
            bottom += d
        ax.grid(False)

        if i == 0:
            # ax.set_ylabel(f'M{minimax}')
            ax.set_ylabel(f'Games')
            ax.text(-13, 25, f'M{minimax}', va='center', ha='left')
        if minimax == 2:
            ax.set_title(nn_labels[i])
        else:
            ax.set_xlabel('Iterations')
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=3, fancybox=True)
fig.savefig('/tmp/nn_ablation.png')
# plt.show()

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
fig.subplots_adjust(bottom=0.3)
ax.plot(range(len(nn_labels)), timing, '-o')
ax.set_xticks(np.arange(len(nn_labels)))
ax.set_xticklabels(nn_labels, rotation=15)
ax.set_xlabel('NN Structure')
ax.set_ylabel('Time [s]')
fig.savefig('/tmp/nn_ablation_time.png')
