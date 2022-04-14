#!/usr/bin/env python
#
# Created on: 2022-03-11
#
import time
from pathlib import Path

import numpy as np
import tqdm
from collections import Counter

from alphazero.utilities import histogram_to_string
from game.board import BoardStateWithMovesCounter
from scripts.utils import evaluation_arg_parser, get_player_from_args

parser = evaluation_arg_parser()
parser.add_argument('num_games', help="number of games", type=int)
parser.add_argument('-results_file', help="where to append the results", type=str, default=None)
parser.add_argument('-note', help="where to append the results", type=str, default=None)
args = parser.parse_args()

players = [get_player_from_args(args, 'white'), get_player_from_args(args, 'black')]

white_rewards_histogram = Counter()
computation_time = np.zeros(2)
action_evaluations = np.zeros(2, dtype=np.int64)
tbar = tqdm.trange(args.num_games)
for igame in tbar:
    players[0].reset()
    players[1].reset()
    s = BoardStateWithMovesCounter()
    for i in tqdm.trange(s.MAXIMUM_MOVES, leave=False):
        current_player = players[1 if s.is_inverted else 0]
        start_time = time.time()
        a = current_player.get_action(s)
        computation_time[1 if s.is_inverted else 0] += time.time() - start_time
        action_evaluations[1 if s.is_inverted else 0] += 1
        s = s.take_action(a).invert()
        if s.is_terminal:
            white_reward = s.reward * (-1 if s.is_inverted else 1)
            white_rewards_histogram[white_reward] += 1
            break
    tbar.set_description_str(
        f'White [{players[0]}] vs Black [{players[1]}]: {histogram_to_string(white_rewards_histogram)}'
    )

avg_time = computation_time / action_evaluations

if args.results_file is not None:
    write_header = not Path(args.results_file).exists()
    with open(args.results_file, 'a') as f:
        if write_header:
            f.write(
                f'White player,Black player,'
                f'num_white_wins,num_white_almost_wins,num_draws,num_black_almost_wins,num_black_wins,'
                f'avg_time_white,avg_time_black,note\n'
            )
        f.write(
            f'{players[0]},{players[1]},{histogram_to_string(white_rewards_histogram, separator=",")},'
            f'{avg_time[0]},{avg_time[1]},{args.note if args.note is not None else ""}\n'
        )
else:
    print(f'White [{players[0]}] vs Black [{players[1]}]: {histogram_to_string(white_rewards_histogram)}')
    print(f'Avg. computation time {avg_time[0]:.5f}s vs {avg_time[1]:.5f}s')
