import Arena
from MCTS import MCTS
from mazebattle.MazeBattleGame import MazeBattleGame, display
from mazebattle.MazeBattlePlayers import *
from mazebattle.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'epsilon': 0.25,
    'dirAlpha': 0.03,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = MazeBattleGame(7)

# all players
rp = HumanMazeBattlePlayer(g).play
# rp2 = RandomPlayer(g).play

# nnet players
# n1 = NNet(g)
# n1.load_checkpoint('./pretrained_models/othello/keras/', '6x6 checkpoint_145.pth.tar')
# args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
# mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('./temp/', 'best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(rp, n2p, g, display=display)
print(arena.playGames(2, verbose=True))
