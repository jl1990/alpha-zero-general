import hashlib
import sys

import numpy as np

sys.setrecursionlimit(100000)


class Node:

    def __init__(self, mcts, board, ident, player):
        self.board = board
        self.edges = []
        self.player = player
        self.mcts = mcts
        self.id = ident
        self.valids = None

    def getValids(self):
        if self.valids is None:
            self.valids = self.mcts.game.getValidMoves(self.board, self.player)
        return self.valids

    def isLeaf(self):
        return len(self.edges) == 0

    def isRootNode(self):
        return self.mcts.root.id == self.id

    def addEdge(self, outNode, prior, action):
        self.edges.append(Edge(self, outNode, prior, action))

    def solveLeaf(self):
        if self not in self.mcts.Es:
            self.mcts.Es[self.id] = self.mcts.game.getGameEnded(self.board, self.player)
        if self.mcts.Es[self.id] != 0:
            return self.mcts.Es[self.id]  # terminal node
        probs, v = self.mcts.nnet.predict(self.board)
        probs = probs * self.getValids()  # masking invalid moves
        sum_Ps_s = np.sum(probs)
        probs /= sum_Ps_s
        for a in [x for x in range(self.mcts.game.getActionSize()) if self.getValids()[x]]:
            next_s, next_player = self.mcts.game.getNextState(self.board, self.player, a)
            stateId = hashlib.md5(
                (str(self.mcts.game.stringRepresentation(next_s)) + str(next_player)).encode('utf-8')).hexdigest()
            if stateId in self.mcts.tree:
                node = self.mcts.tree[stateId]
            else:
                node = Node(self.mcts, next_s, stateId, next_player)
                self.mcts.addNode(node, stateId)
            self.addEdge(node, probs[a], a)
        return v

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
        backfillEdges = []
        while not currentNode.isLeaf():
            cur_best = -float('inf')
            allBest = []
            epsilon = self.mcts.args.epsilon
            if currentNode.isRootNode() and epsilon > 0:
                noise = np.random.dirichlet([self.mcts.args.dirAlpha] * len(currentNode.edges))
            else:
                epsilon = 0
                noise = [0] * len(currentNode.edges)
            ns = sum(edge.stats['N'] for edge in currentNode.edges)
            for idx, edge in enumerate(currentNode.edges):
                U = self.mcts.args.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * noise[idx]) * \
                    np.sqrt(ns) / (1 + edge.stats['N'])
                score = edge.stats['Q'] + U
                if score > cur_best:
                    cur_best = score
                    del allBest[:]
                    allBest.append(edge)
                elif score == cur_best:
                    allBest.append(edge)
            selectedEdge = np.random.choice(allBest)
            selectedEdge.stats['N'] += 1
            currentNode = selectedEdge.outNode
            backfillEdges.append(selectedEdge)
        value = currentNode.solveLeaf()
        for edge in backfillEdges:
            direction = 1 if edge.inNode.player == self.player else -1
            edge.stats['W'] += value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']


class Edge:

    def __init__(self, inNode, outNode, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.action = action
        self.id = inNode.id + '|' + outNode.id

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

        edgesInformation = dict([(edge.action, edge.stats['N']) for edge in edges])
        counts = [edgesInformation.get(i) if i in edgesInformation.keys() else 0 for i in
                  range(self.game.getActionSize())]

        if temp == 0:
            probs = [0] * len(self.game.getActionSize())
            probs[np.random.choice(np.where(np.array(counts) == max(counts))[0])] = 1
            return probs

        probs = [pow(x, 1 / temp) for x in counts]
        sumProbs = np.sum(probs)
        probs = [x / sumProbs for x in probs]
        return probs

    def search(self, canonicalBoard):
        ident = hashlib.md5((str(self.game.stringRepresentation(canonicalBoard)) + '1').encode('utf-8')).hexdigest()
        if ident in self.tree:
            currentNode = self.tree[ident]
        else:
            currentNode = Node(self, canonicalBoard, ident, 1)
            self.addNode(currentNode, ident)
        self.root = currentNode
        currentNode.expand()
        return currentNode

    def addNode(self, node, identifier):
        self.tree[identifier] = node
