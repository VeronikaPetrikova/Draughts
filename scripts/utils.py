#!/usr/bin/env python
#
# Created on: 2022-03-11
#
import argparse
import time
from pathlib import Path

from alphazero.nn import NNPolicy
from evaluation.players import *


def evaluation_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-white', help="Who plays white", type=str, default='nn')
    parser.add_argument('-black', help="Who plays black", type=str, default='nn')

    parser.add_argument('-mcts_num_sims', help="Number of mcts simulation per action", type=int, default=25)
    parser.add_argument('-mcts_cpuct', help="Exploration/exploitation parameter", type=float, default=1.)
    parser.add_argument('-white_mcts_num_sims', help="Number of mcts simulation per action", type=int, default=None)
    parser.add_argument('-black_mcts_num_sims', help="Number of mcts simulation per action", type=int, default=None)
    parser.add_argument('--wait_for_nn', dest='wait_for_nn', action='store_true')

    parser.add_argument('-nn_num_channels', type=int, default=512)
    parser.add_argument('-nn_lin_size', type=int, default=512)

    return parser


def get_mcts_sims(args, color):
    if color == 'white':
        return args.mcts_num_sims if args.white_mcts_num_sims is None else args.white_mcts_num_sims
    else:
        return args.mcts_num_sims if args.black_mcts_num_sims is None else args.black_mcts_num_sims


def get_player_from_args(args, color='white') -> GeneralPlayer:
    """ From parsed arguments build a general player that correspond to the parameterization in the arguments. """
    player = args.white if color == 'white' else args.black
    if player == 'random':
        return RandomPlayer()
    elif player.startswith('minimax'):
        n = int(player[len('minimax'):])
        return MinimaxPlayer(depth=n)
    file = Path(player)
    been_waiting = False
    while not file.exists():
        been_waiting = True
        if args.wait_for_nn:
            print('.', end='', flush=True)
            time.sleep(5.)
        else:
            raise NotImplementedError('File for NN does not exists.')
    if been_waiting:
        time.sleep(60.)  # workaround to not read incomplete files
    nn = NNPolicy(num_channels=args.nn_num_channels, lin_size=args.nn_lin_size)
    nn.load_checkpoint(file.parent, file.name)
    return NNPlayer(nn=nn, num_simulations=get_mcts_sims(args, color), c_puct=args.mcts_cpuct)
