#!/usr/bin/env python
#
# Created on: 2022-03-9
#

import numpy as np
from collections import defaultdict

from game.board import BoardStateWithMovesCounter


class MonteCarloTreeSearch:

    def __init__(self, nn, num_simulations, c_puct=1., dirichlet_alpha=None):
        self.nn = nn
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # controls the exploration
        self.dirichlet_alpha = dirichlet_alpha

        self.Qsa = defaultdict(lambda: defaultdict(lambda: 0))  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(lambda: defaultdict(lambda: 0))  # stores #times edge s,a was visited
        self.Ns = defaultdict(lambda: 0)  # stores #times board s was visited
        self.Psa = {}  # stores initial policy (returned by neural net)

    def reset_search_tree(self):
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Psa.clear()

    def ucb(self, s, a, psa):
        return self.Qsa[s][a] + self.c_puct * psa * np.sqrt(self.Ns[s] + 1e-8) / (1 + self.Nsa[s][a])

    def search(self, s: BoardStateWithMovesCounter, dirichlet_noise=False):
        """
        Perform search in mcts. Stops either at terminal state (game end) or if unvisited node is found.
        """
        if s.is_terminal:
            self.nn(None)
            return -s.reward

        if s not in self.Psa:
            self.Psa[s], v = self.nn(s)
            return -v

        if dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(self.Psa[s]))
            psa = {a: 0.75 * p + 0.25 * n for (a, p), n in zip(self.Psa[s].items(), noise)}
        a = max(s.actions(), key=lambda ai: self.ucb(s, ai, psa=psa[ai] if dirichlet_noise else self.Psa[s][ai]))

        sp = s.take_action(a).invert()  # take action and invert so sp is from next player perspective
        v = self.search(sp)

        self.Qsa[s][a] = (self.Nsa[s][a] * self.Qsa[s][a] + v) / (self.Nsa[s][a] + 1)
        self.Nsa[s][a] += 1
        self.Ns[s] += 1
        return -v

    def get_actions_with_probabilities(self, s: BoardStateWithMovesCounter, temp=0):
        """ Return dictionaries where key is an action and value is a probability of win when taking that action. """
        for i in range(self.num_simulations):
            self.search(s, dirichlet_noise=(i == 0 and self.dirichlet_alpha is not None))

        counts = np.asarray([self.Nsa[s][a] for a in s.actions()])

        if temp == 0:
            """ Return one for one of the value that is maximum and zero elsewhere. """
            random_best_ind = np.random.choice(np.argwhere(counts == np.max(counts)).flatten())
            return {a: 1. if i == random_best_ind else 0. for i, a in enumerate(s.actions())}

        counts = np.power(counts, 1. / temp)
        counts_normalized = counts / np.sum(counts)
        return dict(zip(s.actions(), counts_normalized))

    def get_best_action(self, s: BoardStateWithMovesCounter, temp=0):
        """ Return best actions based on the probabilities computed by counting in mcts. """
        actions = self.get_actions_with_probabilities(s, temp=temp)
        return max(actions.keys(), key=lambda a: actions[a])
