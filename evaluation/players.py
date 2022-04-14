#!/usr/bin/env python
#
# Created on: 2022-03-11
#
import numpy as np

from alphazero.mcts import MonteCarloTreeSearch
from game.board import BoardStateWithMovesCounter


class GeneralPlayer:
    def reset(self):
        """ Method called at the beginning of the game."""
        pass

    def get_action(self, s):
        """ Get action to play for the given state. """
        raise NotImplementedError('Player not fully supported yet.')


class RandomPlayer(GeneralPlayer):
    def get_action(self, s):
        return np.random.choice(s.actions())

    def __repr__(self) -> str:
        return 'Random'


class NNPlayer(GeneralPlayer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mcts = MonteCarloTreeSearch(**kwargs)

    def reset(self):
        self.mcts.reset_search_tree()

    def get_action(self, s):
        return self.mcts.get_best_action(s)

    def __repr__(self) -> str:
        return f'NN{self.mcts.num_simulations}'


class MinimaxPlayer(GeneralPlayer):
    def __init__(self, depth) -> None:
        super().__init__()
        self.depth = depth

    def minimax(self, board: BoardStateWithMovesCounter, depth: int, maximizing: bool = True, first_iteration=True):
        actions = board.actions()
        if len(actions) == 0:
            return -1 if maximizing else 1
        if depth == 0:
            return (board.number_of_my_pieces - board.number_of_opponent_pieces) * (1 if maximizing else -1) / 8

        if maximizing:
            values = np.array([self.minimax(board.take_action(a).invert(), depth - 1, False, False) for a in actions])
            if first_iteration:
                maximum_indicies = np.argwhere(values >= np.max(values))
                return actions[np.random.choice(maximum_indicies.ravel())]
            value = np.max(values)
            return value
        else:
            values = np.array([self.minimax(board.take_action(a).invert(), depth - 1, True, False) for a in actions])
            return np.min(values)

    def get_action(self, s):
        return self.minimax(s, depth=self.depth)

    def __repr__(self) -> str:
        return f'MiniMax{self.depth}'
