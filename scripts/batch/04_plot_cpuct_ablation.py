#!/usr/bin/env python
#
# Created on: 2022-03-31
#

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

all_num_sims = [128] #2, 4, 8, 16, 32, 64,
all_cpucts = [1, 0.5, 0.1, 0.01, 0.001, 2, 5, 10]

ind = np.argsort(all_cpucts)
ind = ind[[0, 1, 2, 4, 5, 6, 7]] # remove 0.5
all_cpucts = np.array(all_cpucts)[ind]

a_texts = [f'NN{v}' for v in all_num_sims]
b_texts = [f'c_puct {v}' for v in all_cpucts]

a, b = len(all_num_sims), 8
wins_white, wins_black = np.nan * np.zeros((a, b)), np.nan * np.zeros((a, b))
losses_white, losses_black = np.nan * np.zeros((a, b)), np.nan * np.zeros((a, b))
draws_white, draws_black = np.nan * np.zeros((a, b)), np.nan * np.zeros((a, b))

for i in range(a):
    file1a = f'../../data/nn_ablation_128_5/progress_nn{all_num_sims[i]}_vs_minimax4.csv'
    file1b = f'../../data/nn_ablation_128_5/cpuct2_nn{all_num_sims[i]}_vs_minimax4.csv'
    file2a = f'../../data/nn_ablation_128_5/progress_minimax4_vs_nn{all_num_sims[i]}.csv'
    file2b = f'../../data/nn_ablation_128_5/cpuct2_minimax4_vs_nn{all_num_sims[i]}.csv'

    df1 = pd.concat([pd.read_csv(file1a), pd.read_csv(file1b)])
    df2 = pd.concat([pd.read_csv(file2a), pd.read_csv(file2b)])

    wins_white[i] = df1['num_white_wins']
    losses_white[i] = df1['num_black_wins']
    draws_white[i] = df1['num_draws']

    wins_black[i] = df2['num_black_wins']
    losses_black[i] = df2['num_white_wins']
    draws_black[i] = df2['num_draws']

wins_white = wins_white[:, ind]
losses_white = losses_white[:, ind]
draws_white = draws_white[:, ind]
wins_black = wins_black[:, ind]
losses_black = losses_black[:, ind]
draws_black = draws_black[:, ind]

a, b = len(all_num_sims), len(all_cpucts)

fig: plt.Figure
fig, axes = plt.subplots(a, b, squeeze=False, sharex=False, sharey=False, figsize=(2.1 * b, 1.5 * a))
fig.subplots_adjust(left=0.045, right=1., top=0.9, bottom=0.025, wspace=0.1, hspace=0.6)

for ax in axes.ravel():
    ax.axis(False)

cred = 'lightcoral'
cdraw = 'silver'
fs = 10
h = 2
n = 25
r = h / 2
dc = 1
dt = 0.5
i, j = 0, 1
for i in range(a):
    for j in range(b):
        ax: plt.Axes = axes[i, j]

        y = 0
        ax.text(0, y + 0.5 * h + dt, f'W:{100 * wins_white[i, j] / n:.0f}%', fontsize=fs, va='bottom', ha='left')
        ax.text(n / 2 - 3, y + 0.5 * h + dt, f'D:{100 * draws_white[i, j] / n:.0f}%', fontsize=fs, va='bottom',
                ha='left')
        ax.text(n, y + 0.5 * h + dt, f'L:{100 * losses_white[i, j] / n:.0f}%', fontsize=fs, va='bottom', ha='right')
        ax.barh(y, wins_white[i, j], color="tab:green", height=h)
        ax.barh(y, draws_white[i, j], left=wins_white[i, j], color=cdraw, height=h)
        ax.barh(y, losses_white[i, j], left=wins_white[i, j] + draws_white[i, j], color=cred, height=h)
        ax.add_artist(Circle((-r - dc, y), radius=r, fc='white', ec='black'))

        y = -h - 0.3
        ax.text(0, y - 0.5 * h - dt, f'W:{100 * wins_black[i, j] / n:.0f}%', fontsize=fs, va='top', ha='left')
        ax.text(n / 2 - 3, y - 0.5 * h - dt, f'D:{100 * draws_black[i, j] / n:.0f}%', fontsize=fs, va='top', ha='left')
        ax.text(n, y - 0.5 * h - dt, f'L:{100 * losses_black[i, j] / n:.0f}%', fontsize=fs, va='top', ha='right')
        ax.barh(y, wins_black[i, j], color="tab:green", height=h)
        ax.barh(y, draws_black[i, j], left=wins_black[i, j], color=cdraw, height=h)
        ax.barh(y, losses_black[i, j], left=wins_black[i, j] + draws_black[i, j], color=cred, height=h)
        ax.add_artist(Circle((-r - dc, y), radius=r, fc='black', ec='black'))

        ax.set_xlim(-2 * r - 2 * dc, n)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.axis(False)

        if j == 0:
            ax.text(-13, (-h - dc) / 2, a_texts[i], fontsize=fs + 2, va='center', ha='left')
        if i == 0:
            ax.text(n / 2, 7, b_texts[j], fontsize=fs + 2, va='center', ha='center')

fig.savefig('/tmp/nn_cpuct_ablation.png')
exit(1)
