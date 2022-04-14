#!/usr/bin/env python
#
# Created on: 2022-03-8
# This class describes board state of the game

import itertools
import numpy as np
from typing import List
from functools import lru_cache

from .action import Action


class BoardStateWithMovesCounter:
    MAXIMUM_MOVES = 75

    def __init__(self, np_board=None, moves_counter=0, is_inverted=False) -> None:
        super().__init__()
        """ Board that is always from white player perspective and have one more column for counter. """
        if np_board is None:
            self._board = self._init_np_board()
        else:
            assert np_board.dtype == np.int8
            self._board = np_board
        self._board.flags.writeable = False
        self.moves_counter = moves_counter
        self.is_inverted = is_inverted

    def copy(self):
        """ Return copy of the board. """
        return BoardStateWithMovesCounter(self._board.copy(), self.moves_counter, self.is_inverted)

    @staticmethod
    def _init_np_board():
        """ Return initial board state """
        b = np.zeros([8, 8], dtype=np.int8)
        b[1::2, 0] = -1
        b[::2, 1] = -1
        b[::2, -1] = 1
        b[1::2, -2] = 1
        return b

    def clear(self):
        """ Clear the board by setting all elements of the board to zero. """
        return BoardStateWithMovesCounter(0 * self._board)

    def invert(self):
        """ Invert the board to the next player view. """
        return BoardStateWithMovesCounter(-np.flip(self._board), self.moves_counter, not self.is_inverted)

    def __getitem__(self, a):
        """ Get piece at index a. """
        return self._board[a]

    def __setitem__(self, a, v):
        """ Get piece at index a. """
        assert self._board.flags.writeable
        self._board[a] = v

    def __hash__(self):
        """ Hash that encodes np array of board but not the counter. """
        return hash(self._board.tobytes())

    def __eq__(self, other):
        return (self._board == other._board).all()

    @property
    def number_of_my_pieces(self):
        return np.sum(self._board > 0)

    @property
    def number_of_opponent_pieces(self):
        return np.sum(self._board < 0)

    def actions(self) -> List[Action]:
        """ Get all actions available for the positive player. """
        return _get_actions(self)

    def take_action(self, action: Action):
        """ Apply action on the board and increase move counter. """
        b = self._board.copy()
        b[action.gx, action.gy] = b[action.sx, action.sy]
        b[action.sx, action.sy] = 0
        for ex, ey in action.remove_route:
            b[ex, ey] = 0
        b[b[:, 0] == 1, 0] = 2
        return BoardStateWithMovesCounter(b, self.moves_counter + 1, self.is_inverted)

    @property
    def is_terminal(self):
        """ Return true if the board state is terminal, i.e. there is no next action or max counts reached. """
        return self.moves_counter >= self.MAXIMUM_MOVES or len(self.actions()) == 0

    @property
    def reward(self):
        """ Return reward of the board. It is -1 if I lost. Other value based on the draw outcome. """
        if len(self.actions()) == 0:
            return -1
        if self.moves_counter >= self.MAXIMUM_MOVES:
            return 0
        return None

    def __repr__(self) -> str:
        return np.array_str(self._board)


@lru_cache(maxsize=None)
def _get_actions(cb: BoardStateWithMovesCounter) -> List[Action]:
    actions = []

    "1) Check possible jumps "
    for i, j in itertools.product(range(8), range(8)):
        if j > 1 and cb[i, j] == 1:
            _jump_routes_pawn(i, j, actions=actions, board=cb)
        elif cb[i, j] == 2:
            _jump_routes_queen(i, j, actions=actions, board=cb)
    if len(actions) > 0:
        return _filter_equal_actions(actions)

    "2) Get non-jumps actions, i.e. just move "
    for i, j in itertools.product(range(8), range(8)):
        if j > 0 and cb[i, j] == 1:
            if i > 0 and cb[i - 1, j - 1] == 0:
                actions.append(Action(i, j, i - 1, j - 1))
            if i < 7 and cb[i + 1, j - 1] == 0:
                actions.append(Action(i, j, i + 1, j - 1))
        elif cb[i, j] == 2:
            for dx, dy in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
                for o in range(1, 8):
                    if not _in_range(i + dx * o, j + dy * o) or cb[i + dx * o, j + dy * o] != 0:
                        break
                    actions.append(Action(i, j, i + dx * o, j + dy * o))
    return _filter_equal_actions(actions)


def _filter_equal_actions(actions: List[Action]):
    """ Remove all duplicates that have same start-end position but might have different remove_routes.
    Keep largest. """
    filtered_list = []
    existing_actions_id = {}
    for a in actions:
        if a in existing_actions_id:
            fid = existing_actions_id[a]
            if len(filtered_list[fid].remove_route) < len(a.remove_route):
                filtered_list[fid] = a.clone()
        else:
            filtered_list.append(a.clone())
            existing_actions_id[a.clone()] = len(filtered_list) - 1
    return filtered_list


def _in_range(x, y):
    return 0 <= x < 8 and 0 <= y < 8


def _jump_routes_pawn(cx, cy, has_jumped=False, actions: List[Action] = None, next_action: Action = None, board=None):
    can_jump_l = cx > 1 and cy > 1 and board[cx - 1, cy - 1] < 0 and board[cx - 2, cy - 2] == 0
    can_jump_r = cx < 6 and cy > 1 and board[cx + 1, cy - 1] < 0 and board[cx + 2, cy - 2] == 0

    if not (can_jump_l or can_jump_r):
        if has_jumped:
            actions.append(next_action.clone_with_new_goal(cx, cy))
            return
        return  # did not jumped yet and has no chance to jump
    if next_action is None:
        next_action = Action(cx, cy)
    if can_jump_l:
        next_action.append_to_remove_route(cx - 1, cy - 1)
        _jump_routes_pawn(
            cx - 2, cy - 2, has_jumped=True, actions=actions, next_action=next_action, board=board
        )
        next_action.remove_route.pop()
    if can_jump_r:
        next_action.append_to_remove_route(cx + 1, cy - 1)
        _jump_routes_pawn(
            cx + 2, cy - 2, has_jumped=True, actions=actions, next_action=next_action, board=board
        )
        next_action.remove_route.pop()


def _jump_routes_queen(cx, cy, actions: List[Action] = None, next_action: Action = None, board=None):
    if next_action is None:
        next_action = Action(cx, cy)

    for dx, dy in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
        for off in range(1, 8):
            if not _in_range(cx + dx * (off + 1), cy + dy * (off + 1)):
                break
            if board[cx + dx * off, cy + dy * off] > 0:
                break  # found my own piece, do not search further more
            if board[cx + dx * off, cy + dy * off] < 0:
                if board[cx + dx * (off + 1), cy + dy * (off + 1)] == 0:
                    jx = cx + dx * (off + 1)
                    jy = cy + dy * (off + 1)
                    jdx = dx
                    jdy = dy
                    next_action.append_to_remove_route(cx + dx * off, cy + dy * off)
                    _queen_check_jumps_from_direction(jx, jy, jdx, jdy, actions, next_action, board)
                    next_action.remove_route.pop()
                break  # cannot jump over him


def _queen_check_jumps_from_direction(jx, jy, jdx, jdy, actions: List[Action] = None, next_action: Action = None,
                                      board=None):
    """ Check if there is continuation of the jump from position jx, jy if jumped from given direction jdx, jdy """
    found_possible_jump = False
    for o in range(8):  # check jump forward in the direction
        if not _in_range(jx + jdx * (o + 1), jy + jdy * (o + 1)):
            break
        if board[jx + jdx * (o + 0), jy + jdy * (o + 0)] > 0:
            break  # found my own piece, do not search further more

        if board[jx + jdx * (o + 0), jy + jdy * (o + 0)] < 0:
            if (jx + jdx * (o + 0), jy + jdy * (o + 0)) in next_action.remove_route:
                break
            if board[jx + jdx * (o + 1), jy + jdy * (o + 1)] == 0:
                found_possible_jump = True
                next_action.append_to_remove_route(jx + jdx * (o + 0), jy + jdy * (o + 0))
                _queen_check_jumps_from_direction(
                    jx + jdx * (o + 1), jy + jdy * (o + 1), jdx, jdy, actions, next_action, board
                )
                next_action.remove_route.pop()
            break

    for o in range(8):  # check perpendicular directions for each empty position in a given direction
        if not _in_range(jx + jdx * o, jy + jdy * o) or board[jx + jdx * o, jy + jdy * o] != 0:
            break
        x, y = jx + jdx * o, jy + jdy * o
        for pdx, pdy in [(-jdx, jdy), (jdx, -jdy)]:
            for p in range(1, 8):
                if not _in_range(x + pdx * (p + 1), y + pdy * (p + 1)):
                    break
                if board[x + pdx * p, y + pdy * p] > 0:
                    break
                if board[x + pdx * p, y + pdy * p] < 0:
                    if (x + pdx * p, y + pdy * p) in next_action.remove_route:
                        break
                    if board[x + pdx * (p + 1), y + pdy * (p + 1)] == 0:
                        found_possible_jump = True
                        next_action.append_to_remove_route(x + pdx * p, y + pdy * p)
                        _queen_check_jumps_from_direction(
                            x + pdx * (p + 1), y + pdy * (p + 1), pdx, pdy, actions, next_action, board
                        )
                        next_action.remove_route.pop()
                    break

    if not found_possible_jump:  # cannot continue, let's add all actions
        for o in range(8):
            if not _in_range(jx + jdx * o, jy + jdy * o):
                break
            if board[jx + jdx * o, jy + jdy * o] != 0:
                break  # found some piece, do not search further more
            actions.append(next_action.clone_with_new_goal(jx + jdx * o, jy + jdy * o))
