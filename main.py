from Coach import Coach
from gobang.GobangGame import GobangGame as Game
from gobang.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 20000,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 10,
    'cpuct': 1.2,
    'epsilon': 0.25,
    'dirAlpha': 0.03,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
