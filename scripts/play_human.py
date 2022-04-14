#!/usr/bin/env python
#
# Created on: 2022-04-7
#
from alphazero.nn import NNPolicy
from evaluation.players import NNPlayer, RandomPlayer, MinimaxPlayer
from gui.board_gui import BoardGUI

"NN player with given number of simulations"
nn = NNPolicy(num_channels=256, lin_size=1024)
folder = '../data/nn_ablation_128_5'
nn.load_checkpoint(folder=folder, filename=f'training_19.pth.tar')
player = NNPlayer(nn=nn, num_simulations=16, c_puct=1.)

"Minimax player with given depth "
# player = MinimaxPlayer(depth=2)

"Random player"
# player = RandomPlayer()

gui = BoardGUI(opponent=player, human_is_white=False)
gui.loop()
