#!/usr/bin/env python
#
# Created on: 2022-03-11
#
import tqdm
import numpy as np
from copy import deepcopy
from collections import deque, Counter

from .mcts import MonteCarloTreeSearch
from .nn import NNPolicy
from game.board import BoardStateWithMovesCounter
from .utilities import histogram_to_string


class SelfPlay:

    def __init__(self, nn: NNPolicy, mcts_num_simulations: int, mcts_c_puct: float = 1., dir_alpha: float = None,
                 temp_threshold: int = 15, maximum_train_examples: int = int(1e6), max_game_iterations=1000) -> None:
        self.mcts = MonteCarloTreeSearch(nn=nn, num_simulations=mcts_num_simulations, c_puct=mcts_c_puct,
                                         dirichlet_alpha=dir_alpha)
        self.temp_threshold = temp_threshold
        self.max_game_iterations = max_game_iterations
        self.train_examples = deque(maxlen=maximum_train_examples)

    def play_one_game(self, run_all_iterations=False, print_progress=False):
        states_actions = []
        s = BoardStateWithMovesCounter()
        complete = False
        trange = tqdm.trange(self.max_game_iterations) if print_progress else range(self.max_game_iterations)
        for i in trange:
            if complete:
                for _ in range(self.mcts.num_simulations):
                    self.mcts.nn(None)
                continue

            actions = self.mcts.get_actions_with_probabilities(s, temp=int(i < self.temp_threshold))
            states_actions.append((s.copy(), deepcopy(actions)))
            a = np.random.choice(list(actions.keys()), p=list(actions.values()))
            s = s.take_action(a).invert()
            if s.is_terminal:
                white_reward = s.reward * (-1 if s.is_inverted else 1)
                states_actions = [(sp, ap, white_reward * (-1 if sp.is_inverted else 1)) for sp, ap in states_actions]
                complete = True
                if not run_all_iterations:
                    break
        assert complete, 'Game must complete'
        return states_actions

    def play_n_games(self, n, reset_search_tree=True, reset_search_tree_every_game=False):
        """ Play N self-play games and append results into the train examples. """
        if reset_search_tree:
            self.mcts.reset_search_tree()
        histogram = Counter()  # track reward histogram, i.e. number of wins for white/black
        tbar = tqdm.trange(n, desc="Self Play")
        for _ in tbar:
            if reset_search_tree_every_game:
                self.mcts.reset_search_tree()
            ex = self.play_one_game()
            self.train_examples.extend(ex)
            histogram[ex[0][-1]] += 1
            tbar.set_description_str('Self Play ' + histogram_to_string(histogram))
