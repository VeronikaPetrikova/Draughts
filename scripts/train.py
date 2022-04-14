#!/usr/bin/env python
#
# Created on: 2022-03-10
#
import time
import json
import shutil
import argparse
from pathlib import Path
from game.board import BoardStateWithMovesCounter
from alphazero.self_play import SelfPlay
from alphazero.nn import NNPolicy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_self_play', help="Number of self play games per training iteration", type=int, default=64)
    parser.add_argument('-mcts_num_sims', help="Number of mcts simulation per action", type=int, default=64)
    parser.add_argument('-mcts_cpuct', help="Exploration/exploitation parameter", type=float, default=1.)
    parser.add_argument('-lr', help="Learning rate", type=float, default=1e-4)
    parser.add_argument('-train_bs', help="Training batch size", type=int, default=1024)
    parser.add_argument('--train_only_batch_on_gpu', dest='train_only_batch_on_gpu', action='store_true')
    parser.add_argument('-dir_alpha', help="Dirichlet noise for exploration.", type=float, default=0.3)

    parser.add_argument('-num_iterations', help="Number training iterations", type=int, default=20)
    parser.add_argument('-start_from_iteration', type=int, default=0)
    parser.add_argument('-folder', help="Folder to store the results into.", type=str, default=None)
    parser.add_argument('--clear_folder', dest='clear_folder', action='store_true')

    parser.add_argument('-nn_num_channels', type=int, default=512)
    parser.add_argument('-nn_lin_size', type=int, default=512)

    options = parser.parse_args()
    folder_name = options.folder if options.folder is not None else time.strftime("%Y%m%d_%H%M%S")
    folder = Path(__file__).absolute().parent.parent.joinpath('data').joinpath(folder_name)
    if options.clear_folder and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Saving data into folder {folder}')
    with open(folder.joinpath('commandline_args.json'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    nn = NNPolicy(num_channels=options.nn_num_channels, lin_size=options.nn_lin_size)
    self_play = SelfPlay(nn, mcts_num_simulations=options.mcts_num_sims,
                         mcts_c_puct=options.mcts_cpuct, dir_alpha=options.dir_alpha,
                         max_game_iterations=BoardStateWithMovesCounter.MAXIMUM_MOVES,
                         maximum_train_examples=100000)
    if options.start_from_iteration > 0:
        print(f'Loading checkpoint: training_{options.start_from_iteration}.pth.tar')
        nn.load_checkpoint(folder=folder, filename=f'training_{options.start_from_iteration}.pth.tar')
    else:
        nn.save_checkpoint(folder=folder, filename=f'training_0.pth.tar')
    for i in range(options.start_from_iteration + 1, options.num_iterations):
        print(f'Training iteration: {i}')
        self_play.play_n_games(options.num_self_play, reset_search_tree_every_game=options.mcts_reset_every_game)

        nn.train(self_play.train_examples, batch_size=options.train_bs, lr=options.lr,
                 all_data_on_gpu=not options.train_only_batch_on_gpu)
        print(f'Saving checkpoint')
        nn.save_checkpoint(folder=folder, filename=f'training_{i}.pth.tar')
        # nn.save_checkpoint(folder=folder, filename='best.pth.tar')

