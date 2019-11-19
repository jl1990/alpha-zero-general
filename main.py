from Coach import Coach
from mazebattle.MazeBattleGame import MazeBattleGame as Game
from mazebattle.keras.NNet import NNetWrapper as nn
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
    'load_model': True,
    'load_folder_file': ('./temp', 'checkpoint_11.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = Game(7)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
