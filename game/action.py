#!/usr/bin/env python
#
# Created on: 2022-03-10
#

from __future__ import annotations
from typing import List, Tuple

import numpy as np


class Action:
    """
    Single action representation. Contains start and goal position of the piece and list of positions that should be
    cleared after action is taken.
    Two actions are equal iff theirs start and goal positions are equal regardless of the remove route.
    """

    def __init__(self, sx=-1, sy=-1, gx=-1, gy=-1, remove_route: List[Tuple[int, int]] = None) -> None:
        """ Create action with given parameters. Remove route is duplicated internally."""
        self.sx, self.sy, self.gx, self.gy = sx, sy, gx, gy
        self.remove_route = [] if remove_route is None else remove_route.copy()

    def __hash__(self):
        return hash((self.sx, self.sy, self.gx, self.gy))

    def __eq__(self, other: Action):
        return self.sx == other.sx and self.sy == other.sy and self.gx == other.gx and self.gy == other.gy

    def append_to_remove_route(self, ex: int, ey: int):
        self.remove_route.append((ex, ey))

    def clone(self) -> Action:
        return Action(self.sx, self.sy, self.gx, self.gy, self.remove_route)

    def clone_with_new_goal(self, gx, gy) -> Action:
        return Action(self.sx, self.sy, gx, gy, self.remove_route)

    @property
    def index_in_nn_output(self):
        return np.ravel_multi_index([self.sx, self.sy, self.gx, self.gy], dims=[8] * 4)

    def __repr__(self) -> str:
        return f'({self.sx},{self.sy}) -> ({self.gx},{self.gy}) with {len(self.remove_route)} removals'
