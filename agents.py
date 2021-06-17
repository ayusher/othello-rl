from monte_carlo_tree_search import MCTS#, Node
import os
import time, random
import numpy as np
from multiprocessing import Lock, Pool, Process, Manager
from multiprocessing.managers import BaseManager
import multiprocessing
from collections import defaultdict, deque
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K

class self_play_agent:
    def __init__(self, t=10):
        self.tree = MCTS()
        self.time_constraint = t

        # determines who is likely to win
        self.value = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (5, 5), input_shape=(8, 8, 1,), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        # determines probability of making moves
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, (3, 3), input_shape=(8, 8, 1,), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(1, (1, 1), activation = "linear")
        ])

        if "value" in os.listdir() and "policy" in os.listdir():
            self.value = tf.keras.models.load_model("value", compile=False)
            self.policy = tf.keras.models.load_model("policy", compile=False)

        self.value.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=[tf.keras.metrics.BinaryAccuracy()])
        self.policy.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        self.action = None

    def reset_tree(self):
        self.tree = MCTS()

    def search(self, board, t=0):
        count = 0
        start = time.time()
        obs = board.mat.reshape((8, 8, 1))
        if t==0:
            while (time.time()-start)<self.time_constraint:
                self.tree.do_deep_rollout(board, self.value, self.policy)
                count+=1
                #print((board.mat.reshape((8, 8, 1)) == obs).all())
            print("rollouts {}".format(count))
        else:
            while count<25:
                self.tree.do_deep_rollout(board, self.value, self.policy, count==0)
                #self.tree.exploration_weight = max(.999999*self.tree.exploration_weight, 1)
                count+=1

        #obs = board.mat.reshape((8, 8, 1))
        if t==0: print("value network output (predicted chance of winning): {:.2f}".format(self.value.predict(np.array([obs]))[0][0]))
        probs, self.action = self.tree._get_action_prob(board, t)
        #print("rollouts {}".format(count))

        if probs is not None:
            return obs, probs
        else: return None, None

    def move(self, board):
        #return self.tree.choose(board)
        if board.legal_moves(board.next_turn)[-1]==1: self.action = 64
        return board.make_move(board.next_turn, self.action)

class mobility_alphabeta_agent:
    def __init__(self, t=10):
        self.time_constraint = t
        self.max_depth = 3
        self.terminal = False

    def search(self, board):
        pass

    def weighted(self, board):
        weights = np.array([120, -20, 20, 5, 5, 20, -20, 120, -20, -40, -5, -5, -5, -5, -40, -20, 20, -5, 15, 3, 3, 15, -5, 20, 5, -5, 3, 3, 3, 3, -5, 5, 5, -5, 3, 3, 3, 3, -5, 5, 20, -5, 15, 3, 3, 15, -5, 20, -20, -40, -5, -5, -5, -5, -40, -20, 120, -20, 20, 5, 5, 20, -20, 120]).reshape((8, 8))
        black, white = 0, 0
        for row in range(8):
            for col in range(8):
                if board.black_array2d[row][col]==1: black += weights[row][col]
                elif board.white_array2d[row][col]==1: white += weights[row][col]
        if board.next_turn == 1: return black-white
        else: return white-black

    def mobility(self, board):
        #return board.score()
        boardcop = type(board)(board.black, board.white, board.opp[board.next_turn])
        c, c2 = len(board.find_children()), len(boardcop.find_children())
        return (c-c2)

    def eval(self, board):
        if board.get_winner()==board.next_turn: return 1
        elif board.get_winner()==None: return 0
        else: return -1

    def move(self, board):
        self.max_depth=3
        self.terminal = False
        start = time.time()
        outsav = (0, board.find_random_child())
        while(True):
            #print(outsav)
            out = self.alphabeta(board, start)
            if type(out[1])==int and out == (0, 0):
                out = outsav
                break
            else: outsav = out
            self.max_depth+=1
            if self.terminal: break
        #print(self.terminal)
        if self.terminal: print("terminal")
        else: print("max depth", self.max_depth)
        return out[1]

    def alphabeta(self, board, start, depth=0, alpha=-999, beta=999):
        #depth+=1
        if time.time()>start+self.time_constraint: return 0, 0
        if depth>self.max_depth:
            return self.mobility(board), None
        elif board.is_terminal():
            return self.eval(board), None
        children = board.find_children()
        bestchild = children[0]
        for child in children:
            if alpha>=beta: break
            val = self.alphabeta(child, start, depth+1, -beta, -alpha)
            if val==(0,0): return (0,0)
            else: val = -val[0]
            if val>alpha:
                alpha = val
                bestchild = child
        return alpha, bestchild

class weighted_alphabeta_agent:
    def __init__(self, t=10):
        self.time_constraint = t
        self.max_depth = 3
        self.terminal = False

    def search(self, board):
        pass

    def weighted(self, board):
        weights = np.array([120, -20, 20, 5, 5, 20, -20, 120, -20, -40, -5, -5, -5, -5, -40, -20, 20, -5, 15, 3, 3, 15, -5, 20, 5, -5, 3, 3, 3, 3, -5, 5, 5, -5, 3, 3, 3, 3, -5, 5, 20, -5, 15, 3, 3, 15, -5, 20, -20, -40, -5, -5, -5, -5, -40, -20, 120, -20, 20, 5, 5, 20, -20, 120]).reshape((8, 8))
        black, white = 0, 0
        for row in range(8):
            for col in range(8):
                if board.black_array2d[row][col]==1: black += weights[row][col]
                elif board.white_array2d[row][col]==1: white += weights[row][col]
        if board.next_turn == 1: return black-white
        else: return white-black

    def mobility(self, board):
        #return board.score()
        boardcop = type(board)(board.black, board.white, board.opp[board.next_turn])
        c, c2 = len(board.find_children()), len(boardcop.find_children())
        return (c-c2)

    def eval(self, board):
        if board.get_winner()==board.next_turn: return 1
        elif board.get_winner()==None: return 0
        else: return -1

    def move(self, board):
        self.max_depth=3
        self.terminal = False
        start = time.time()
        outsav = (0, board.find_random_child())
        while(True):
            #print(outsav)
            out = self.alphabeta(board, start)
            if type(out[1])==int and out == (0, 0):
                out = outsav
                break
            else: outsav = out
            self.max_depth+=1
            if self.terminal: break
        #print(self.terminal)
        if self.terminal: print("terminal")
        else: print("max depth", self.max_depth)
        return out[1]

    def alphabeta(self, board, start, depth=0, alpha=-999, beta=999):
        #depth+=1
        if time.time()>start+self.time_constraint: return 0, 0
        if depth>self.max_depth:
            return self.weighted(board), None
        elif board.is_terminal():
            return self.eval(board), None
        children = board.find_children()
        bestchild = children[0]
        for child in children:
            if alpha>=beta: break
            val = self.alphabeta(child, start, depth+1, -beta, -alpha)
            if val==(0,0): return (0,0)
            else: val = -val[0]
            if val>alpha:
                alpha = val
                bestchild = child
        return alpha, bestchild

class MCTS_agent:
    def __init__(self, t=10):
        self.tree = MCTS()
        self.time_constraint = t

    def search(self, board):
        count = 0
        start = time.time()
        while (time.time()-start)<self.time_constraint:
            self.tree.do_rollout(board)
            count+=1
            #print(count, end="\r")
        #print(len(self.tree.children))
        print("rollouts {}".format(count))

    def move(self, board):
        return self.tree.choose(board)


class random_agent:
    def __init__(self, t=10):
        pass

    def search(self, board):
        pass

    def move(self, board):
        return board.find_random_child()
