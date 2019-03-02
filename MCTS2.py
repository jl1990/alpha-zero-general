import math
import sys

import numpy as np

sys.setrecursionlimit(100000)


class Node:

    def __init__(self, game, board, epsilon, dirAlpha, identifier=0):
        self.board = board
        self.edges = {}
        self.id = identifier
        self.dirAlpha = dirAlpha
        self.valids = self.game.getValidMoves(board, 1)
        self.epsilon = epsilon
        self.game = game

    def isLeaf(self):
        return len(self.edges) == 0

    def isRootNode(self):
        return id == 0

    def addEdge(self, outNode, prior, action):
        self.edges.add(Edge(self, outNode, prior, action))

    def solveLeaf(self):
        self.Ps[s], v = self.nnet.predict(canonicalBoard)
        self.Ps[s] = self.Ps[s] * self.valids  # masking invalid moves
        sum_Ps_s = np.sum(self.Ps[s])
        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s  # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            print("All valid moves were masked, do workaround.")
            self.Ps[s] = self.Ps[s] + self.valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Ns[s] = 0
        return -v

    def expand(self):
        if self.isLeaf():
            return self.solveLeaf()

        cur_best = -float('inf')
        # best_act = -1
        allBest = []

        # Add Dirichlet noise for root node if needed.
        useDirichletNoise = self.isRootNode() and self.epsilon > 0
        if useDirichletNoise:
            noise = np.random.dirichlet([self.dirAlpha] * len(self.valids))

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if self.valids[a]:
                if (s, a) in self.Qsa:
                    q = self.Qsa[(s, a)]
                    n_s_a = self.Nsa[(s, a)]
                    # u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    q = 0
                    n_s_a = 0
                    # u = np.random.random_sample() + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                p = self.Ps[s][a]
                if useDirichletNoise:
                    p = (1 - self.epsilon) * p + self.epsilon * noise[a]

                u = q + self.args.cpuct * p * math.sqrt(self.Ns[s]) / (1 + n_s_a)

                if u > cur_best:
                    cur_best = u
                    # best_act = a
                    del allBest[:]
                    allBest.append(a)
                elif u == cur_best:
                    allBest.append(a)

        a = np.random.choice(allBest)

        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, False)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


class Edge:

    def __init__(self, inNode, outNode, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.action = action
        self.id = inNode.state.id + '|' + outNode.state.id

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }

    def __eq__(self, other):
        return self.id == other.id


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, True)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            maxi = max(counts)
            allBest = np.where(np.array(counts) == maxi)[0]
            bestA = np.random.choice(allBest)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        countSum = sum(counts)
        if countSum == 0:
            # Random choice?
            probs = [0] * len(counts)
            probs[np.random.randint(0, len(probs))] = 1
        else:
            probs = [x / float(countSum) for x in counts]
        return probs

    def search(self, canonicalBoard, isRootNode):
        # Need to create node from this status
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        currentNode = Node(self.game, canonicalBoard, self.args.epsilon, self.args.dirAlpha)
        return currentNode.expand()
