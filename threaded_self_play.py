#import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import pickle

import binascii
#import neat
import numpy as np
import random
import threading
from multiprocessing import Process, Lock
import multiprocessing
import time

from agents import self_play_agent
from collections import Counter
import othello
from othello import Othello
import tensorflow as tf
from tensorflow.keras import backend as K
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
multiprocessing.set_start_method('spawn', force=True)
#tf.config.run_functions_eagerly(True)
#physical_devices = tf.config.list_physical_devices('GPU')
#for dev in physical_devices:
#    tf.config.experimental.set_memory_growth(dev, True)

def custom_loss(y_true, y_pred):
    #a2, b2 = a.numpy(), b.numpy()

    #return tf.compat.v1.losses.softmax_cross_entropy(a, b)
    #return tf.nn.softmax_cross_entropy_with_logits(a, b)
    #difference between true label and predicted label
    e = tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred)
    out = K.mean(e)
    return out


def pit(agents, winlist, first):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf

    files = agents[2:]
    agents = agents[:2]

    agents[0].value = tf.keras.models.load_model(files[0], compile=False)
    agents[0].policy = tf.keras.models.load_model(files[1], compile=False)

    agents[1].value = tf.keras.models.load_model(files[2], compile=False)
    agents[1].policy = tf.keras.models.load_model(files[3], compile=False)

    count, turn = 0, 1
    board = Othello()
    temperature = 1e-8
    while True:
        count += 1
        agent = agents[turn-1]
        badd, policyadd = agent.search(board, temperature)
        board = agent.move(board)
        if board.is_terminal():
            break

        if turn==1: turn=2
        elif turn==2: turn=1

    winner = board.get_winner()
    if winner == first:
        winlist.append(1)
    elif winner is not None:
        winlist.append(0)

def play_game(shared_list, agents, lock, e):
    temperature = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf

    files = agents[2:]
    agents = agents[:2]

    agents[0].value = tf.keras.models.load_model(files[0])
    agents[0].policy = tf.keras.models.load_model(files[1], compile=False)

    agents[1].value = tf.keras.models.load_model(files[2])
    agents[1].policy = tf.keras.models.load_model(files[3], compile=False)

    bs, wins, policy = [], [], []
    board = Othello()
    #fnameadd = binascii.b2a_hex(os.urandom(15)).decode('utf-8')+".txt"
    #fname="{}/{}".format(e, fnameadd)
    fname = ""
    board.print(fname)
    turn = 1
    count = 0
    while True:
        count += 1
        if count>8: temperature = 1e-8
        agent = agents[turn-1]
        badd, policyadd = agent.search(board, temperature)

        if badd is not None and policyadd is not None:
            bs.append(badd)
            policy.append(policyadd)
            wins.append(turn)

        board = agent.move(board)

        board.print(fname)
        if board.is_terminal():
            break

        if turn==1: turn=2
        elif turn==2: turn=1

    winner = board.get_winner()
    if winner == None: return None

    wins = [1 if i==winner else 0 for i in wins]

    lock.acquire()
    for i in range(len(bs)):
        shared_list[0].append(bs[i])
        shared_list[1].append(wins[i])
        shared_list[2].append(policy[i])
    lock.release()
    return winner-1

if __name__=="__main__":
    if len(sys.argv)>1: ti = float(sys.argv[1])
    else: ti = 10

    agent1 = self_play_agent(ti)
    agent2 = self_play_agent(ti)

    agent1.value.set_weights(agent2.value.get_weights())
    agent1.policy.set_weights(agent2.policy.get_weights())

    print("value")
    agent1.value.summary()

    print("policy")
    agent1.policy.summary()

    manager = multiprocessing.Manager()
    shared_list = manager.list([manager.list() for _ in range(3)])
    #temp = 1
    trainx, policyy, valuey = [], [], []
    if "trainx.npy" in os.listdir() and "policyy.npy" in os.listdir() and "valuey.npy" in os.listdir():
        trainx = np.load('trainx.npy').tolist()
        policyy = np.load('policyy.npy').tolist()
        valuey = np.load('valuey.npy').tolist()

    agent2.policy.save("policy")
    agent2.value.save("value")

    for e in range(1000):
        print("episode: {}".format(e))

        agent1.policy, agent1.value = None, None
        agent2.policy, agent2.value = None, None

        jobs = []
        gamecount = 0
        while gamecount < 64:
            lock = Lock()
            for _ in range(16):
                jobs.append(Process(target=play_game, args=(shared_list, [agent1, agent2, "value", "policy", "value", "policy"], lock, e)))
                jobs.append(Process(target=play_game, args=(shared_list, [agent2, agent1, "value", "policy", "value", "policy"], lock, e)))
                gamecount += 2
                if gamecount >= 64: break

            for j in jobs:
                j.start()

            for j in jobs:
                j.join()

            jobs = []

        agent2.policy = tf.keras.models.load_model("policy", compile=False)
        agent2.value = tf.keras.models.load_model("value", compile=False)

        agent2.value.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=[tf.keras.metrics.BinaryAccuracy()])
        agent2.policy.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=custom_loss, metrics=["mae"])

        #print("temperature: {}, memory size: {}".format(temp, len(shared_list[0])))
        boardlist = [shared_list[0][g] for g in range(len(shared_list[0]))]
        policylist = [shared_list[2][g] for g in range(len(shared_list[0]))]
        valuelist = [shared_list[1][g] for g in range(len(shared_list[0]))]

        for i in range(len(boardlist)):
            rbs = [boardlist[i], np.rot90(boardlist[i]), np.rot90(boardlist[i], 2), np.rot90(boardlist[i], 3)]
            rps = [policylist[i], np.rot90(policylist[i]), np.rot90(policylist[i], 2), np.rot90(policylist[i], 3)]
            for j in range(4):
                rbs.append(np.fliplr(rbs[j]))
                rps.append(np.fliplr(rps[j]))
            trainx = trainx + rbs
            policyy = policyy + rps
            valuey = valuey + [valuelist[i] for _ in range(8)]

        memsize = 400000
        if len(trainx)>memsize:
            trainx, policyy, valuey = trainx[-memsize:], policyy[-memsize:], valuey[-memsize:]

        shared_list = manager.list([manager.list() for _ in range(3)])
        print("memory size: {}".format(len(trainx)))

        with open('trainx.npy', 'wb') as f:
            np.save(f, np.array(trainx))

        with open('policyy.npy', 'wb') as f:
            np.save(f, np.array(policyy))

        with open('valuey.npy', 'wb') as f:
            np.save(f, np.array(valuey))

        print("policy fit")
        agent2.policy.fit(np.array(trainx), np.array(policyy), epochs = 3, batch_size=128)

        print("value fit")
        agent2.value.fit(np.array(trainx), np.array(valuey), epochs = 3, batch_size=128)

        agent2.policy.save("policy")
        agent2.value.save("value")
        print("models saved")

        agent1.reset_tree()
        agent2.reset_tree()
