#import tensorflow as tf

import os
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

from agents import MCTS_agent, random_agent, mobility_alphabeta_agent, self_play_agent, weighted_alphabeta_agent
from collections import Counter
import othello
from othello import Othello

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def play_game(l, agents):
    board = Othello()
    #fname = "logs/"+binascii.b2a_hex(os.urandom(15)).decode('utf-8')+".txt"
    fname=""
    board.print()
    turn = 1
    while True:
        agent = agents[turn-1]
        lm = board.legal_moves(turn)
        print(board.map[turn], type(agent).__name__, [x for x in range(65) if lm[x]==1])
        agent.search(board)
        board = agent.move(board)

        board.print()
        if board.is_terminal():
            break

        if turn==1: turn=2
        elif turn==2: turn=1

    winner = board.get_winner()
    if winner == None: l.append("tie")
    else: l.append(type(agents[board.get_winner()-1]).__name__+str(winner))


if __name__=="__main__":
    '''
    jobs = []

    manager = multiprocessing.Manager()
    return_list = manager.list()

    for _ in range(multiprocessing.cpu_count()):
        jobs.append(Process(target=play_game, args=(return_list, [minimax_agent(), MCTS_agent()], )))

    for j in jobs: j.start()
    for j in jobs: j.join()

    d = dict(Counter(return_list))
    total = sum(list(d.values()))
    for k in d:
        d[k]=d[k]/total
        print("{} win rate: {:.2f}%".format(k, 100*d[k]))

    print(d)
    '''
    if len(sys.argv)>1: ti = float(sys.argv[1])
    else: ti = 10
    outcome = []
    players = [MCTS_agent, random_agent, mobility_alphabeta_agent, weighted_alphabeta_agent, self_play_agent]
    for p in range(len(players)):
        print("[{}] {}".format(p, players[p]))

    p1 = int(input("black "))
    p2 = int(input("white "))

    play_game(outcome, [players[p1](ti), players[p2](ti)])
    print(outcome)


