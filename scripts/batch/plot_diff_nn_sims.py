#!/usr/bin/env python
#
# Created on: 2022-03-15
#
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_folders = ['c', 'd', 'e', 'f']
train_sims = [25, 50, 100, 100]
eval_sims = [25, 50, 100]

fig: plt.Figure
fig, axes = plt.subplots(len(train_sims), len(eval_sims), squeeze=False, sharey=True,  sharex=True, figsize=(6.4 * 2, 4.8))

for i in range(len(train_sims)):
    for j in range(len(eval_sims)):
        print(f'{i}, {j}')
        ax: plt.Axes = axes[i, j]
        file1 = f'../../data/{train_folders[i]}/progress_minimax2_vs_nn{eval_sims[j]}.csv'
        if not Path(file1).exists():
            continue
        # file2 = f'../../data/{train_folders[i]}/progress_nn{eval_sims[j]}_vs_minimax2.csv'
        if j == 0:
            ax.set_ylabel(f'Train sims: {train_sims[i]}')
        if i == len(train_sims) - 1:
            ax.set_xlabel(f'Eval sims: {eval_sims[j]}')

        df1 = pd.read_csv(file1)
        # df2 = pd.read_csv(file2)

        labels = df1['White player'].astype(str) + ' vs ' + df1['Black player'].astype(str) + ' ' + df1['note'].astype(
            str)

        toplot = ['num_white_wins', 'num_draws', 'num_black_wins']

        my_cmap = plt.cm.get_cmap('copper_r')
        bottom = None

        for k, color in zip(toplot, my_cmap(np.linspace(0, 1, 3))):
            if bottom is None:
                bottom = 0 * df1[k]
            # ax.bar(labels.values, df1[k], bottom=bottom, color=color)
            ax.bar(range(len(df1[k])), df1[k], bottom=bottom, color=color)
            bottom += df1[k]
        # ax.grid(False)

        #
        # for k in range(len(toplot)):
        #     if bottom is None:
        #         bottom = 0 * df1[toplot[k]]
        #     both = df1[toplot[k]] + df2[toplot[2 - k]]
        #     ax.bar(labels.values, both, label=k if i == 0 and j == 0 else None,
        #            bottom=bottom, color=my_cmap(k / len(toplot)))
        #     bottom += both
        #
        # ax.tick_params(axis='x', labelrotation=90)
        # # ax.set_title(file)
        # ax.grid(False)

    # for toploti, k in enumerate(toplot):
    #         if bottom is None:
    #             bottom = 0 * df1[k]
    #         ax.bar(labels.values, df[k], label=k if i == 0 else None, bottom=bottom, color=my_cmap(toploti / len(toplot)))
    #         bottom += df[k]
    #     ax.tick_params(axis='x', labelrotation=90)
    #     ax.set_title(file)
    #     ax.grid(False)

# fig.subplots_adjust(bottom=0.3)
# for i, file in enumerate(files):
#     ax: plt.Axes = axes[0, i]
#     df = pd.read_csv(file)
#     labels = df['White player'].astype(str) + ' vs ' + df['Black player'].astype(str) + ' ' + df['note'].astype(str)
#     bottom = None
#     toplot = ['num_white_wins', 'num_white_almost_wins', 'num_draws', 'num_black_almost_wins', 'num_black_wins']
#     my_cmap = plt.cm.get_cmap('copper_r')
#     for j, k in enumerate(toplot):
#         if bottom is None:
#             bottom = 0 * df[k]
#         ax.bar(labels.values, df[k], label=k if i == 0 else None, bottom=bottom, color=my_cmap(j / len(toplot)))
#         bottom += df[k]
#     ax.tick_params(axis='x', labelrotation=90)
#     ax.set_title(file)
#     ax.grid(False)
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), ncol=5, fancybox=True)

plt.show()
