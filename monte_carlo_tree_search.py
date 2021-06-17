"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import math
import time
import random

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.predicted_value = {}
        self.predicted_policy = {}

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        #print(node.board)
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q.get(n) / self.N.get(n)  # average reward

        return max(self.children.get(node), key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def do_deep_rollout(self, node, value, policy, firstsim = False):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._deep_select(node, value, policy, firstsim)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._deep_simulate(leaf, value, policy)
        self._backpropagate(path, reward)

    def _deep_select(self, node, value, policy, firstsim = False):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            #if len(self.N)<10: print(self.N)
            #node.print()
            path.append(node)
            if node not in self.children.keys() or node.is_terminal():
                # node is either unexplored or terminal
                return path
            #print(type(self.children.keys()))
            unexplored = set(self.children.get(node)) - set(self.children.keys())
            #print(unexplored, self.children.keys())
            if len(unexplored)>0:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._deep_uct_select(node, value, policy, firstsim)  # descend a layer deeper

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            #if len(self.N)<10: print(self.N)
            #node.print()
            path.append(node)
            if node not in self.children.keys() or node.is_terminal():
                # node is either unexplored or terminal
                return path
            #print(type(self.children.keys()))
            unexplored = set(self.children.get(node)) - set(self.children.keys())
            #print(unexplored, self.children.keys())
            if len(unexplored)>0:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children.keys():
            return  # already expanded
        temp = {node: node.find_children()}
        self.children.update(temp)
        #self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            #print(node.board[2])
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            #node.print()
            invert_reward = not invert_reward

    def _deep_simulate(self, node, value, policy):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            #print(node.board[2])
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward

            if node not in self.children.keys():
                if node in self.predicted_value: reward = self.predicted_value[node]
                else:
                    obs = node.mat.reshape(8, 8, 1)
                    reward = value(np.array([obs]), training=False).numpy()[0][0]
                    self.predicted_value[node] = reward
                return 1 - reward if invert_reward else reward

            '''
            tmp = 0
            bd, act = self._get_action_prob(node)
            cs, ays = node.find_children_tuple()
            node = cs[ays.index(act)]
            #node.print()
            '''
            node = self._deep_uct_select(node, value, policy)
            invert_reward = not invert_reward

    def _deep_simulate2(self, node, value, policy):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True

        if node.is_terminal():
            reward = node.reward()
            return 1 - reward if invert_reward else reward

        obs = node.mat.reshape(8, 8, 1)

        reward = value.predict(np.array([obs]))[0][0]
        return 1 - reward if invert_reward else reward

    def _get_action_prob(self, node, t=1):
        tup = node.find_children_tuple()
        nums = {}
        for i in range(len(tup[0])):
            nums[tup[1][i]] = self.N.get(tup[0][i], 0)

        s = sum(list(nums.values()))

        if list(nums.keys())==[64]: return None, 64
        if s==0:
            #print(len(nums.keys()))
            s=len(list(nums.keys()))
            for key in nums:
                nums[key]=1

        arr = np.array([0. for _ in range(64)])
        for k, v in nums.items():
            #print(k)
            arr[k]=v/s
            #print(v, s)

        arr = arr/np.sum(arr)
        #print(parr)
        action = np.random.choice(len(arr), p=arr)
        if t<=1e-8:
            #print("temp 0")
            action = max(list(range(len(arr))), key = lambda x: arr[x])
            arr = np.array([0. for _ in range(64)])
            arr[action] = 1
            return arr.reshape((8, 8, 1)), action
        #print("action", action)
        return arr.reshape((8, 8, 1)), action


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            #print(self.N[node])
            #print("here")
            self.N.update({node: self.N.get(node, 0)+1})
            self.Q.update({node: self.Q.get(node, 0)+reward})
            #self.N[node] += 1
            #self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        #print("here")
        # All children of node should already be expanded:
        assert all(n in self.children.keys() for n in self.children.get(node))

        #print(type(self.N))
        log_N_vertex = math.log(self.N.get(node))

        #time.sleep(.1)
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q.get(n) / self.N.get(n) + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N.get(n)
            )

        return max(self.children.get(node), key=uct)

    def _deep_uct_select(self, node, value, policy, isroot = False):
        "Select a child of node, balancing exploration & exploitation"
        #print("here")
        # All children of node should already be expanded:
        #assert all(n in self.children.keys() for n in self.children.get(node))

        #print(type(self.N))
        #log_N_vertex = math.log(self.N.get(node))

        def uct(n, p):
            "Upper confidence bound for trees"
            return self.Q.get(n, .5) + self.exploration_weight * p * math.sqrt(
                self.N.get(n, 0)) / (1 + self.N.get(n, 0))

        #time.sleep(.1)
        if node in self.predicted_policy: out = self.predicted_policy[node]
        else:
            obs = node.mat.reshape(8, 8, 1)
            #start = time.time()
            out = policy(np.array([obs]), training=False).numpy()[0].flatten().tolist()
            #print(time.time()-start)
            out = softmax(out)
            #print(out, sum(out))
            self.predicted_policy[node] = out

        if isroot:
            for i in range(len(out)):
                if out[i]>0: out[i] = out[i]*.75 + random.random()*.25

        tup = node.find_children_tuple()
        if tup[1] == [64]: return tup[0][0]

        ps = [out[tup[1][x]] for x in range(len(tup[1]))]
        comp = list(zip(tup[0], ps))
        comp = [(i[0], uct(i[0], i[1])) for i in comp]
        #print(out, sum(out))
        return max(comp, key=lambda x: x[1])[0]

    def _deep_uct_select2(self, node, value, policy):
        "Select a child of node, balancing exploration & exploitation"
        #print("here")
        # All children of node should already be expanded:
        assert all(n in self.children.keys() for n in self.children.get(node))

        #print(type(self.N))
        log_N_vertex = math.log(self.N.get(node))

        #time.sleep(.1)
        obs = node.mat.reshape(8, 8, 1)
        out = policy.predict(np.array([obs]))[0].flatten().tolist()

        tup = node.find_children_tuple()
        if tup[1] == [64]: return tup[0][0]
        comp = list(zip(tup[0], [out[tup[1][x]] for x in range(len(tup[1]))]))
        return max(comp, key=lambda x: x[1])[0]

class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
