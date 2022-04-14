#!/usr/bin/env python
#
# Created on: 2022-03-10
#

from abc import ABC
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch import optim

from game.action import Action
from game.board import BoardStateWithMovesCounter


class NNPolicy:

    def __init__(self, evaluation_device=None, **kwargs):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.evaluation_device = self.device if evaluation_device is None else evaluation_device
        self.nnet = NeuralNet(**kwargs).to(self.device)

    def board_states_to_torch(self, states: List[BoardStateWithMovesCounter], device=None):
        boards = np.array([[b == -2, b == -1, b == 1, b == 2] for s in states for b in (s._board,)])  # N x 4 x 8 x 8
        return torch.tensor(boards, dtype=torch.float32, device=self.device if device is None else device)

    @staticmethod
    def compute_actions_with_probabilities_from_nn_output(state: BoardStateWithMovesCounter, nn_prob):
        """
        From given array of length 64*64, computes dictionary of actions with associated probabilities.
        Policy output is probability of all possible actions.
        Functions here defines transformation between the board actions probabilities and the policy action probabilities.
        Policy output space is represented by (8*8)*(8*8) probabilities indicating start and goal positions.
        Board action probability is dictionary of action and associated probability.
        """
        d = {a: nn_prob[a.index_in_nn_output] for a in state.actions()}
        sum_p = sum(d.values())
        if sum_p > 0:
            return {a: p / sum_p for a, p in d.items()}
        return {a: 1. / len(d) for a, p in d.items()}  # equal probability if wrong nn prediction

    @staticmethod
    def compute_nn_output_from_actions_with_probabilities(actions_probabilities: Dict[Action, float]):
        nn_prob = np.zeros(64 * 64)
        for a, p in actions_probabilities.items():
            nn_prob[a.index_in_nn_output] = p
        return nn_prob

    def examples_to_tensors(self, examples, device=None):
        device = self.device if device is None else device
        all_boards = self.board_states_to_torch([e[0] for e in examples], device=device)
        all_pis = torch.tensor(
            np.array([self.compute_nn_output_from_actions_with_probabilities(e[1]) for e in examples]),
            dtype=torch.float32, device=device
        )
        all_values = torch.tensor(np.array([e[2] for e in examples]), dtype=torch.float32, device=device)
        return all_boards, all_pis, all_values

    def __call__(self, s: BoardStateWithMovesCounter):
        """ Predicts action probabilities and value based on the board state """
        if s is None:
            return None
        self.nnet.eval()
        with torch.no_grad():
            self.nnet.to(self.evaluation_device)
            log_pi, v = self.nnet(self.board_states_to_torch([s], device=self.evaluation_device))
        prob = torch.exp(log_pi).detach().cpu().numpy()[0]
        actions_with_prob = self.compute_actions_with_probabilities_from_nn_output(s, prob)
        return actions_with_prob, v.detach().cpu().numpy()[0]

    def eval_batch_of_boards(self, states):
        self.nnet.eval()
        with torch.no_grad():
            self.nnet.to(self.evaluation_device)
            log_pi, v = self.nnet(self.board_states_to_torch(states, device=self.evaluation_device))
        prob = torch.exp(log_pi).detach().cpu().numpy()
        all_actions_probs = [
            self.compute_actions_with_probabilities_from_nn_output(s, p) for s, p in zip(states, prob)
        ]
        return all_actions_probs, v.detach().cpu().numpy()[:, 0].tolist()

    def train(self, examples, epochs=1, w_l2_norm=1e-6, batch_size=128, lr=1e-3, all_data_on_gpu=True):
        """
        examples: list of examples, each example is of form (board_states, target_pi, target_v)
        """
        print(f'Training on {len(examples)} examples. ')
        optimizer = optim.Adam(self.nnet.parameters(), weight_decay=w_l2_norm, lr=lr)
        dataset = TensorDataset(*self.examples_to_tensors(examples, device=self.device if all_data_on_gpu else 'cpu'))

        self.nnet.train()
        for epoch in range(epochs):
            for _, bd in enumerate(tqdm.tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))):
                boards, target_pis, target_vs = [d.to(self.device) for d in bd]
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    @staticmethod
    def loss_pi(targets, outputs):
        return -torch.sum(targets * outputs) / targets.shape[0]

    @staticmethod
    def loss_v(targets, outputs):
        return torch.sum((targets - outputs.view(-1)).square()) / targets.shape[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': self.nnet.state_dict(), }, folder.joinpath(filename))

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = Path(folder).joinpath(filename)
        if not filepath.exists():
            raise ("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class NeuralNet(torch.nn.Module, ABC):

    def __init__(self, num_channels=512, lin_size=512):
        super().__init__()
        self.board_x, self.board_y = 8, 8
        self.action_size = 64 * 64
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(4, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels * (self.board_x) * (self.board_y), lin_size)
        self.fc_bn1 = nn.BatchNorm1d(lin_size)
        self.fc2 = nn.Linear(lin_size, lin_size)
        self.fc_bn2 = nn.BatchNorm1d(lin_size)

        self.fc3 = nn.Linear(lin_size, self.action_size)
        self.fc4 = nn.Linear(lin_size, 1)

    def forward(self, s):
        s = s.view(-1, 4, self.board_x, self.board_y)  # batch_size x 4 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x) x (board_y)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x) x (board_y)
        s = s.view(-1, self.num_channels * (self.board_x) * (self.board_y))

        s = F.relu(self.fc_bn1(self.fc1(s)))  # batch_size x lin_size
        s = F.relu(self.fc_bn2(self.fc2(s)))  # batch_size x lin_size

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
