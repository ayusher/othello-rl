import numpy as np
import re

class Othello:
    def __init__(self):
        self.BLACK = 1
        self.WHITE = 2
        self.EMPTY = 0
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3], self.board[4, 4] = self.BLACK, self.BLACK
        self.board[3, 4], self.board[4, 3] = self.WHITE, self.WHITE
        self.opp = {self.BLACK: self.WHITE, self.WHITE: self.BLACK}
        self.map = {self.BLACK: "●", self.WHITE: "○", self.EMPTY:"-"}

        self.rows = [list(range(x*8, x*8+8)) for x in range(8)]
        self.cols = [list(range(x, 64, 8)) for x in range(8)]
        self.diags = [list(range(x, 64, 9)) for x in range(8)]+[list(range(x*8, 64, 9)) for x in range(1, 8)]
        self.diags = self.diags + [list(range(x, 64, 7)) for x in range(8)]+[list(range(x*8+7, 64, 7)) for x in range(1, 8)]
        self.diags = [i for i in self.diags if len(i)>=3]
        self.checks = self.rows+self.cols+self.diags


    def print(self):
        print("  ", end="")
        print(" ".join([str(x) for x in range(8)]))
        for row in range(len(self.board)):
            print(row, end=" ")
            for item in self.board[row]:
                print(self.map[item], end=" ")
            print()


    def make_move(self, action, turn):
        if action in self.legal_moves(turn):
            self.board[action//8, action%8]=turn

        else: return False
        
        board = self.board.flatten()
        for item in self.checks:
            li = [str(board[x]) for x in item]
            s = "".join(li)
            g1 = [(x.start(), x.end()) for x in re.finditer(r"({})({})+{}".format(turn, self.opp[turn], turn), s)]   
            for tup in g1:
                for i in range(*tup):
                    self.board[item[i]//8, item[i]%8] = turn
        

    def legal_moves(self, turn):
        legals = set()
        board = self.board.flatten()
        for item in self.checks:
            li = [str(board[x]) for x in item]
            s = "".join(li)
            g1 = [x.end()-1 for x in re.finditer(r"({})({})+{}".format(turn, self.opp[turn], self.EMPTY), s)]
            g1 = g1+[x.start() for x in re.finditer(r"{}({})+({})".format(self.EMPTY, self.opp[turn], turn), s)]
            
            for i in range(len(g1)):
                legals.add(item[g1[i]])
        return legals

            

