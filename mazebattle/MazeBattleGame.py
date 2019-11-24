# coding=utf-8
from __future__ import print_function

import random

import numpy as np

from Game import Game
from .MazeBattleLogic import Board

"""
Game class implementation for the game of MazeBattleGame.

"""


class MazeBattleGame(Game):
    def __init__(self, n=random.randint(5, 20)):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return self.n, self.n

    def getActionSize(self):
        # return number of actions
        return 24

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n, initialBoard=board.copy())
        actionType = None
        direction = None
        # [move1, .. , move8, build1, .., build8, break1, .., break8, shoot1, .., shoot8]
        if 0 <= action <= 7:
            actionType = Board.ACTION_MOVE
            direction = action + 1
        elif 8 <= action <= 15:
            actionType = Board.ACTION_BUILD_WALL
            direction = action - 7
        elif 16 <= action <= 23:
            actionType = Board.ACTION_BREAK_WALL
            direction = action - 15

        move = (actionType, direction)
        b.execute_move(move, player)
        return b.pieces, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.n, initialBoard=board)
        legalMoves = b.get_legal_moves(player)
        return np.array(legalMoves)

    def getGameEnded(self, board, player, turn):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        b = Board(self.n, initialBoard=board)
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        return 1e-4 if (turn >= self.n * 5) else 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return self.exchange_board(board, player)

    def exchange_board(self, board, color):
        copied = board.copy()
        if color != 1:
            for x in range(self.n):
                for y in range(self.n):
                    if copied[x][y] == 1:
                        copied[x][y] = -1
                    elif copied[x][y] == -1:
                        copied[x][y] = 1
                    elif copied[x][y] == Board.TAG_PLAYER2_STARTING_POINT:
                        copied[x][y] = Board.TAG_PLAYER1_STARTING_POINT
                    elif copied[x][y] == Board.TAG_PLAYER1_STARTING_POINT:
                        copied[x][y] = Board.TAG_PLAYER2_STARTING_POINT
        return copied

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert (len(pi) == self.getActionSize())
        return []

    def stringRepresentation(self, board):
        # nxn numpy array (canonical board)
        return board.tostring()


def display(board):
    n = board.shape[0]
    print()
    for x in range(n):
        for y in range(n):
            tag = board[x][y]
            if tag == Board.TAG_EMPTY:
                print("□", end='')
            elif tag == Board.TAG_WALL_0_HIT:
                print("░", end='')
            elif tag == Board.TAG_WALL_1_HIT:
                print("█", end='')
            elif tag == Board.TAG_PLAYER1_STARTING_POINT:
                print("🏁", end='')
            elif tag == Board.TAG_PLAYER2_STARTING_POINT:
                print("️✪", end='')
            elif tag == Board.TAG_PLAYER1:
                print("⛹", end='')
            elif tag == Board.TAG_PLAYER2:
                print("☠", end='')
            else:
                print("☯", end='')
        print("\n")
    print("--")
