import math
import sys

import numpy as np

sys.setrecursionlimit(100000)


class Node:

    def __init__(self, mcts, board, ident, player):
        self.board = board
        self.edges = {}
        self.player = player
        self.mcts = mcts
        self.id = ident # TODO check if we need to append player
        self.valids = self.mcts.game.getValidMoves(board, 1)
        self.backfillEdges = []

    def isLeaf(self):
        return len(self.edges) == 0

    def isRootNode(self):
        return self.mcts.root.id == self.id

    def addEdge(self, outNode, prior, action):
        self.edges.add(Edge(self, outNode, prior, action))

    def solveLeaf(self):
        if self not in self.mcts.Es:
            self.mcts.Es[self.id] = self.mcts.game.getGameEnded(self.board, 1)
        if self.mcts.Es[self.id] != 0:
            # terminal node
            return -self.mcts.Es[self.id]
        probs, v = self.mcts.nnet.predict(self.board)
        probs = probs * self.valids  # masking invalid moves
        sum_Ps_s = np.sum(probs)
        if sum_Ps_s > 0:
            probs /= sum_Ps_s  # renormalize
        else:
            print("All valid moves were masked, do workaround.")
            probs = probs + self.valids
            probs /= np.sum(probs)
        for a in range(self.mcts.game.getActionSize()):
            if self.valids[a]:
                next_s, next_player = self.mcts.game.getNextState(self.board, 1, a)
                next_s = self.mcts.game.getCanonicalForm(next_s, next_player)
                stateId = self.mcts.game.stringRepresentation(next_s)
                if stateId in self.mcts.tree:
                    node = self.mcts.tree[stateId]
                else:
                    node = Node(self.mcts, next_s, self.mcts.game.stringRepresentation(next_s), next_player)
                    self.mcts.addNode(node, stateId)
                self.addEdge(node, probs[a], a)
        return -v

    def expand(self):
        '''
                self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        :return:
        '''
        currentNode = self
        currentBoard = self.board
        while ~currentNode.isLeaf():
            cur_best = -float('inf')
            allBest = []
            epsilon = self.mcts.args.epsilon
            if self.isRootNode() and epsilon > 0:
                noise = np.random.dirichlet([self.mcts.args.dirAlpha] * len(self.edges))
            else:
                epsilon = 0
                noise = [0] * len(self.edges)
            ns = sum(edge.stats['N'] for edge in self.edges)
            for idx, edge in enumerate(self.edges):
                U = self.mcts.args.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * noise[idx]) * \
                    np.sqrt(ns) / (1 + edge.stats['N'])
                Q = edge.stats['Q']
                score = Q + U
                if score > cur_best:
                    cur_best = score
                    del allBest[:]
                    allBest.append(edge)
                elif score == cur_best:
                    allBest.append(edge)
            selectedEdge = np.random.choice(allBest)
            a = selectedEdge.action
            currentBoard, next_player = self.mcts.game.getNextState(currentBoard, 1, a)
            currentBoard = self.mcts.game.getCanonicalForm(currentBoard, next_player)
            currentNode = selectedEdge.outNode
            self.backfillEdges.append(selectedEdge)
        value = self.solveLeaf()
        currentPlayer = currentNode.player
        for edge in self.backfillEdges:
            direction = 1 if edge.playerTurn == currentPlayer else -1
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']


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
        self.tree = {}
        self.root = None
        self.Es = {}

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        edges = self.root.edges
        counts = [0] * self.game.getActionSize()
        for edge in edges:
            counts[edge.action] = 1

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

    def search(self, canonicalBoard):
        ident = self.game.stringRepresentation(canonicalBoard)
        if ident not in self.tree:
            currentNode = Node(self, canonicalBoard, ident, 1)
            self.addNode(currentNode, ident)
        else:
            currentNode = self.tree[ident]
        self.root = currentNode
        currentNode.expand()
        return currentNode

    def addNode(self, node, identifier):
        self.tree[identifier] = node
