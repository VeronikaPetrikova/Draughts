#!/usr/bin/env python
#
# Created on: 2022-04-7
#
import itertools
from tkinter import *

import numpy as np

from game.action import Action
from game.board import BoardStateWithMovesCounter


class BoardGUI:
    def __init__(self, opponent, human_is_white=True) -> None:
        super().__init__()

        self.opponent = opponent
        self.human_is_white = human_is_white
        self.window = Tk()
        self.window.geometry('800x800')
        self.window.bind('<Escape>', lambda x: self.window.destroy())
        self.window.bind('q', lambda x: self.window.destroy())
        self.game = BoardStateWithMovesCounter()
        self.checker_board = self.create_board()
        self.units = self.create_units()
        if not human_is_white:
            self.take_action(self.opponent.get_action(self.game), inverted=True)
        self.reset_board_color()

    def create_board(self):
        board = np.ndarray((8, 8), dtype=object)
        for i, j in itertools.product(range(8), range(8)):
            f = Frame(self.window, width=100, height=100)
            f.pack_propagate(False)
            f.place(x=100 * i, y=100 * j)
            board[i, j] = f
        return board

    def reset_board_color(self):
        ckb = np.indices((8, 8)).sum(axis=0) % 2
        for i, j in itertools.product(range(8), range(8)):
            clr = 'gray' if ckb[i, j] == 1 else 'white'
            self.checker_board[i, j].configure(background=clr)

        for a in self.game.actions():
            self.checker_board[a.sx, a.sy].configure(background='deep sky blue')

    def update_unit(self, unit, i=None, j=None, txt=None):
        if i is not None and j is not None:
            unit[0].place(x=100 * i + 25, y=100 * j + 25)
        if txt is not None:
            unit[2].set(txt)

    def create_units(self):
        board = BoardStateWithMovesCounter()
        units = np.ndarray((8, 8), dtype=object)
        for i, j in itertools.product(range(8), range(8)):
            if board[i, j] == 1:
                clr = 'white' if self.human_is_white else 'black'
            elif board[i, j] == -1:
                clr = 'black' if self.human_is_white else 'white'
            else:
                continue
            inv_clr = 'white' if clr == 'black' else 'black'
            f = Frame(self.window, width=50, height=50)
            f.pack_propagate(False)
            f.place(x=100 * i, y=100 * j)
            string_var = StringVar()
            l = Label(f, textvariable=string_var, bg=clr, fg=inv_clr)
            l.pack(fill=BOTH, expand=1)
            units[i, j] = (f, l, string_var)
            self.update_unit(units[i, j], i, j, 'p')

            l.bind("<ButtonPress-1>", self.on_start)
            l.bind("<B1-Motion>", self.on_drag)
            l.bind("<ButtonRelease-1>", self.on_drop)
            l.configure(cursor="hand1")
        return units

    def on_start(self, event):
        sx, sy = self.xy_to_ij(event.widget.master.winfo_x(), event.widget.master.winfo_y())
        for a in self.game.actions():
            if a.sx == sx and a.sy == sy:
                self.checker_board[a.gx, a.gy].configure(background='lime green')

    def on_drag(self, event):
        pass

    def on_drop(self, event):
        self.reset_board_color()
        dx, dy = event.x, event.y
        sx, sy = self.xy_to_ij(event.widget.master.winfo_x(), event.widget.master.winfo_y())
        gx, gy = self.xy_to_ij(event.widget.master.winfo_x() + dx, event.widget.master.winfo_y() + dy)

        a = Action(sx=sx, sy=sy, gx=gx, gy=gy)
        if a not in self.game.actions():
            print('Invalid action')
            return

        self.take_action(self.game.actions()[self.game.actions().index(a)])

        if self.game.is_terminal:
            self.print_final('You are awesome and you won!' if self.game.reward < 0 else 'Draw')
            return

        self.take_action(self.opponent.get_action(self.game), inverted=True)

        self.reset_board_color()
        if self.game.is_terminal:
            self.print_final('Looser!' if self.game.reward < 0 else 'Draw')

    def print_final(self, txt=''):
        f = Frame(self.window, width=600, height=300)
        f.pack_propagate(False)
        f.place(x=100, y=200)
        l = Label(f, text=txt, bg='white')
        l.pack(fill=BOTH, expand=1)

    def take_action(self, a: Action, inverted=False):
        self.game = self.game.take_action(a)
        sx, sy, gx, gy = a.sx, a.sy, a.gx, a.gy
        if inverted:
            sx, sy, gx, gy = 7 - sx, 7 - sy, 7 - gx, 7 - gy
        g = self.game if not self.game.is_inverted else self.game.invert()
        if not self.human_is_white:
            g = g.invert()
        self.update_unit(self.units[sx, sy], i=gx, j=gy, txt='Q' if abs(g[gx, gy]) == 2 else None)
        self.units[gx, gy] = self.units[sx, sy]
        self.units[sx, sy] = None
        for (i, j) in a.remove_route:
            if inverted:
                i, j = 7 - i, 7 - j
            print(f'{inverted}, {i}, {j}')
            self.update_unit(self.units[i, j], i=-1, j=-1, txt='kaput')
            self.units[i, j] = None

        self.game = self.game.invert()

    def xy_to_ij(self, x, y):
        return int(x / 100) % 8, int(y / 100) % 8

    def loop(self):
        self.window.mainloop()
