#!/usr/bin/env python
#
# Created on: 2022-03-15
#
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

wins, wins_white, wins_black = np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6))
losses, losses_white, losses_black = np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6))
draws, draws_white, draws_black = np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6)), np.nan * np.zeros((6, 6))
for i in range(6):
    for j in range(0, 6):
        if i == j:
            continue
        file1 = f'../../data/minimax{i + 1}_vs_minimax{j + 1}.csv'
        file2 = f'../../data/minimax{j + 1}_vs_minimax{i + 1}.csv'

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        wins_white[i, j] = df1['num_white_wins']
        losses_white[i, j] = df1['num_black_wins']
        draws_white[i, j] = df1['num_draws']

        wins_black[i, j] = df2['num_black_wins']
        losses_black[i, j] = df2['num_white_wins']
        draws_black[i, j] = df2['num_draws']

        wins[i, j] = df1['num_white_wins'] + df2['num_black_wins']
        losses[i, j] = df1['num_black_wins'] + df2['num_white_wins']
        draws[i, j] = df1['num_draws'] + df2['num_draws']

time_white = np.nan * np.zeros((6, 6))
time_black = np.nan * np.zeros((6, 6))
timeing = np.zeros(6)
for i in range(6):
    for j in range(6):
        if i == j:
            continue
        file1 = f'../../data/minimax{i + 1}_vs_minimax{j + 1}.csv'
        df1 = pd.read_csv(file1)
        timeing[i] += df1['avg_time_white'] / 5
        timeing[j] += df1['avg_time_black'] / 5

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(range(1, 7), timeing, '-o')
ax.set_xlabel('Minimax depth')
ax.set_ylabel('Time [s]')
fig.savefig('/tmp/minimax_avg_time.png')
# plt.show()
# exit(1)

fig: plt.Figure
fig, axes = plt.subplots(6, 6, squeeze=False, sharex=False, sharey=False, figsize=(2.1 * 6, 0.8 * 6))
fig.subplots_adjust(left=0.03, right=1., top=0.9, bottom=0.025, wspace=0.1, hspace=0.6)

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
for i in range(6):
    for j in range(0, 6):
        ax: plt.Axes = axes[i, j]

        if j == 0:
            ax.text(-8, (-h - dc) / 2, f'M{i + 1}', fontsize=fs + 2, va='center', ha='left')
        if i == 0:
            ax.text(n / 2, 7.5 if j == 0 else 6, f'M{j + 1}', fontsize=fs + 2, va='center', ha='center')
        ax.set_xlim(-2 * r - 2 * dc, n)
        ax.set_aspect('equal')

        if i == j:
            continue
        # if j < i:
        #     continue

        y = 0
        ax.text(0, y + 0.5 * h + dt, f'W:{100 * wins_white[i, j] / n:.0f}%', fontsize=fs, va='bottom', ha='left')
        ax.text(n / 2 - 3, y + 0.5 * h + dt, f'D:{100 * draws_white[i, j] / n:.0f}%', fontsize=fs, va='bottom',
                ha='left')
        ax.text(n, y + 0.5 * h + dt, f'L:{100 * losses_white[i, j] / n:.0f}%', fontsize=fs, va='bottom', ha='right')
        ax.barh(y, wins_white[i, j], color="tab:green", height=h)
        ax.barh(y, draws_white[i, j], left=wins_white[i, j], color=cdraw, height=h)
        ax.barh(y, losses_white[i, j], left=wins_white[i, j] + draws_white[i, j], color=cred, height=h)
        ax.add_artist(Circle((-r - dc, y), radius=r, fc='white', ec='black'))

        y = -h - dc
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
#
# xspan = np.linspace(0.1, 1.025, 7)
# xspan = (xspan[:-1] + xspan[1:]) / 2
# yspan = np.linspace(0.9, 0, 7)
# yspan = (yspan[:-1] + yspan[1:]) / 2
# for i in range(6):
#     fig.text(xspan[i], 0.95, f'Minimax {i + 1}', fontsize=fs + 2, ha='center')
#     fig.text(0.05, yspan[i], f'Minimax {i + 1}', fontsize=fs + 2, ha='center', rotation=0)

fig.savefig('/tmp/minimax_vs_minimax.png')
exit(1)

fig, axes = plt.subplots(1, 3, squeeze=True, figsize=(6.4 * 2, 4.8))

ax = axes[0]
ax.imshow(wins, vmin=0, vmax=50)
ax.set_xticklabels(list(range(0, 7)))
ax.set_yticklabels(list(range(0, 7)))
ax.grid(False)

ax = axes[1]
ax.imshow(draws, vmin=0, vmax=50)
ax.set_xticklabels(list(range(0, 7)))
ax.set_yticklabels(list(range(0, 7)))
ax.grid(False)

ax = axes[2]
img = ax.imshow(losses, vmin=0, vmax=50)
ax.set_xticklabels(list(range(0, 7)))
ax.set_yticklabels(list(range(0, 7)))
ax.grid(False)

fig.colorbar(img)

plt.show()
