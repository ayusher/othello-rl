import numpy as np
import re, random
from monte_carlo_tree_search import MCTS, Node

cols = [list(range(x, 64, 8)) for x in range(8)]
rows = [list(range(x*8, x*8+8)) for x in range(8)]
diags = [list(range(x, 64-8*x, 9)) for x in range(8)]+[list(range(x*8, 64, 9)) for x in range(1, 8)]
diags = diags + [list(range(x, 8*(x+1)-1, 7)) for x in range(0, 8)]+[list(range(x*8+7, 64, 7)) for x in range(1, 8)]
diags = [i for i in diags if len(i)>=3]

patterns = [re.compile(r"(?=({}{}+{}))".format(1,2,1)), re.compile(r"(?=({}{}+{}))".format(2,1,2))]
p1s = [re.compile(r"{}{}+{}".format(1,2,0)), re.compile(r"{}{}+{}".format(2,1,0))]
p2s = [re.compile(r"{}{}+{}".format(0,2,1)), re.compile(r"{}{}+{}".format(0,1,2))]

class Othello(Node):
    def __init__(self, board=None, next_turn=1):
        self.next_turn = next_turn
        self.BLACK = 1
        self.WHITE = 2
        self.EMPTY = 0
        if board is None: 
            self.board = np.zeros((8, 8), dtype=int)
            self.board[3, 3], self.board[4, 4] = self.BLACK, self.BLACK
            self.board[3, 4], self.board[4, 3] = self.WHITE, self.WHITE
        else: self.board = board

        self.opp = {self.BLACK: self.WHITE, self.WHITE: self.BLACK}
        self.map = {self.WHITE: "●", self.BLACK: "○", self.EMPTY:"-"}


        self.checks = rows+cols+diags

        self.patterns = patterns

        '''
        self.pattern = re.compile(r"{}{}+{}".format(self.next_turn, self.opp[next_turn], self.next_turn))
        self.p1 = re.compile(r"{}{}+{}".format(self.next_turn, self.opp[self.next_turn], self.EMPTY))
        self.p2 = re.compile(r"{}{}+{}".format(self.EMPTY, self.opp[self.next_turn], self.next_turn))
        '''

        self.ms = self.legal_moves(self.next_turn)
        self.f = self.board.flatten().tolist()
        

    def print(self):
        print()
        print("  ", end="")
        print(" ".join([str(x) for x in range(8)]))
        for row in range(len(self.board)):
            print(row, end=" ")
            for item in self.board[row]:
                print(self.map[item], end=" ")
            print()
        print()

    def make_move(self, action, turn):
        b = self.board.copy()
        if action in self.legal_moves(turn):
            if action==64:
                return b
            b[action//8, action%8]=turn
        else:
            print(action, "FAIL")
            return False

        board = b.flatten()
        for item in self.checks:
            li = [str(board[x]) for x in item]
            s = "".join(li)
            #print(s)
            g1 = [(x.start(1), x.end(1)) for x in self.patterns[turn-1].finditer(s)]
            #print(g1)
            for tup in g1:
                if tup[1]-tup[0]==0: continue
                l = list(range(*tup))
                if item[l[0]]==action or item[l[-1]]==action:
                    for i in range(*tup):
                        b[item[i]//8, item[i]%8] = turn
        
        return b
        
    def legal_moves(self, turn):
        p1 = p1s[turn-1]
        p2 = p2s[turn-1]

        legals = set()
        board = self.board.flatten()
        for item in self.checks:
            li = [str(board[x]) for x in item]
            s = "".join(li)
            g1 = [x.end()-1 for x in p1.finditer(s)]
            g1 = g1+[x.start() for x in p2.finditer(s)]
            
            for i in range(len(g1)):
                legals.add(item[g1[i]])
        if len(legals)>0: return legals
        else: return set([64])

    def find_children(self):
        out = []
        for action in self.ms:
            out.append(Othello(self.make_move(action, self.next_turn), self.opp[self.next_turn]))
        return out
    
    def find_children_tuple(self):
        out = []
        out2 = []
        for action in self.ms:
            out.append(Othello(self.make_move(action, self.next_turn), self.opp[self.next_turn]))
            out2.append(action)
        return out, out2

    def find_random_child(self):
        return Othello(self.make_move(random.choice(list(self.ms)), self.next_turn), self.opp[self.next_turn])
    
    def is_terminal(self):
        if list(self.ms)==[64] and list(self.legal_moves(self.opp[self.next_turn]))==[64]: return True
        if self.f.count(self.WHITE)==0 or self.f.count(self.BLACK)==0: return True
        return self.f.count(self.EMPTY)==0

    def reward(self):
        if self.next_turn == self.get_winner(): return 1
        elif self.get_winner() == None: return .5
        else: return 0

    def get_winner(self):
        if self.f.count(self.BLACK)>self.f.count(self.WHITE): return self.BLACK
        elif self.f.count(self.WHITE)>self.f.count(self.BLACK): return self.WHITE
        else: return None

    def __hash__(self):
        return hash((tuple(self.board.flatten()), self.next_turn))

    def __eq__(node1, node2):
        if type(node1)!=type(node2): return False
        return tuple(node1.f)==tuple(node2.f) and node1.next_turn==node2.next_turn
    
    def step(self, action):
        return Othello(self.make_move(action, self.next_turn), self.opp[self.next_turn])

    def check(self):
        for i in self.checks:
            o = Othello(np.zeros((8, 8), dtype=int))
            for t in i:
                o.board[t//8, t%8]=self.WHITE
            o.print()
