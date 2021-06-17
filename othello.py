import numpy as np
import random

initial_black = np.uint64(0b00010000 << 24 | 0b00001000 << 32)
initial_white = np.uint64(0b00001000 << 24 | 0b00010000 << 32)

class Othello:
    def __init__(self, black=initial_black, white=initial_white, next_turn = 1):
        self.next_turn = next_turn
        self.black = black
        self.white = white
        self.EMPTY = 0
        self.map = {2: "●", 1: "○", 0:"-"}
        self.opp = {1: 2, 2: 1, 0: 0}
        self.black_array2d = bit_to_array(self.black, 64).reshape((8, 8))
        self.white_array2d = bit_to_array(self.white, 64).reshape((8, 8))
        self.children, self.actions = None, None
        if self.next_turn == 1: self.mat = self.black_array2d.astype(int) - self.white_array2d.astype(int)
        else: self.mat = self.white_array2d.astype(int) - self.black_array2d.astype(int)
        #print(self.mat)

    def reward(self):
        winner = self.get_winner()
        if winner is None: return .5
        elif winner == self.next_turn: return 1
        else: return 0
    
    def get_winner(self):
        if self.black_array2d.flatten().tolist().count(1)>self.white_array2d.flatten().tolist().count(1): return 1
        elif self.white_array2d.flatten().tolist().count(1)>self.black_array2d.flatten().tolist().count(1): return 2
        elif self.white_array2d.flatten().tolist().count(1)==self.black_array2d.flatten().tolist().count(1): return None

    def is_terminal(self):
        if self.legal_moves(1).tolist()==[0 for _ in range(64)]+[1] and self.legal_moves(2).tolist()==[0 for _ in range(64)]+[1]:
            return True

    def find_children(self):
        player = self.next_turn
        if self.children is not None: return self.children
        children = []
        lms = self.legal_moves(player)
        for move in range(65):
            if lms[move]==1: children.append(self.make_move(player, move))
        self.children = children
        return children

    def find_random_child(self):
        children = self.find_children()
        return children[int(random.random()*len(children))]

    def find_children_tuple(self):
        player = self.next_turn
        if self.children is not None and self.actions is not None: return self.children, self.actions
        children = []
        actions = []
        lms = self.legal_moves(player)
        for move in range(65):
            if lms[move]==1:
                children.append(self.make_move(player, move))
                actions.append(move)
        self.children = children
        self.actions = actions
        return children, actions

    def print(self, fname="hi"):
        if fname=="": return
        print()
        print("  ", end="")
        print(" ".join([str(x) for x in range(8)]))
        blacker = bit_to_array(self.black, 64).reshape((8, 8))
        whiter = bit_to_array(self.white, 64).reshape((8, 8))
        for row in range(8):
            print(row, end=" ")
            for col in range(8):
                if blacker[row][col]==1: print(self.map[1], end=" ")
                elif whiter[row][col]==1: print(self.map[2], end=" ")
                else: print(self.map[0], end=" ")
            print()
        print()

    def get_own_and_enemy(self, player):
        if player == 1:
            return self.black, self.white
        else:
            return self.white, self.black

    def get_own_and_enemy_array2d(self, player):
        if player == 1:
            return self.black_array2d, self.white_array2d
        else:
            return self.white_array2d, self.black_array2d

    def make_move(self, player, move):
        if move == 64:
            return Othello(self.black, self.white, next_turn = self.opp[self.next_turn])
        bit_move = np.uint64(0b1 << move)
        own, enemy = self.get_own_and_enemy(player)
        flipped_stones = get_flipped_stones_bit(bit_move, own, enemy)
        own |= flipped_stones | bit_move
        enemy &= ~flipped_stones
        if player == 1:
            return Othello(own, enemy, next_turn = self.opp[self.next_turn])
        else:
            return Othello(enemy, own, next_turn = self.opp[self.next_turn])

    def legal_moves(self, player):
        own, enemy = self.get_own_and_enemy(player)
        legal_moves_without_pass = bit_to_array(legal_moves_bit(own, enemy), 64)
        if np.sum(legal_moves_without_pass) == 0:
            return np.concatenate((legal_moves_without_pass, [1]))
        else:
            return np.concatenate((legal_moves_without_pass, [0]))
    
    def __hash__(self):
        return hash((self.black, self.white, self.next_turn))

    def __eq__(node1, node2):
        if type(node1)!=type(node2): return False
        return node1.white == node2.white and node1.black == node2.black and node1.next_turn==node2.next_turn


left_right_mask = np.uint64(0x7e7e7e7e7e7e7e7e)
top_bottom_mask = np.uint64(0x00ffffffffffff00)
corner_mask = left_right_mask & top_bottom_mask


def legal_moves_bit(own, enemy):
    legal_moves = np.uint64(0)
    legal_moves |= search_legal_moves_left(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_left(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(7))
    legal_moves |= search_legal_moves_right(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_right(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(7))
    legal_moves &= ~(own | enemy)
    return legal_moves


def search_legal_moves_left(own, enemy, mask, offset):
    return search_contiguous_stones_left(own, enemy, mask, offset) >> offset


def search_legal_moves_right(own, enemy, mask, offset):
    return search_contiguous_stones_right(own, enemy, mask, offset) << offset


def get_flipped_stones_bit(bit_move, own, enemy):
    flipped_stones = np.uint64(0)
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(7))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(7))
    return flipped_stones


def search_flipped_stones_left(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_left(bit_move, enemy, mask, offset)
    if own & (flipped_stones >> offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_flipped_stones_right(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_right(bit_move, enemy, mask, offset)
    if own & (flipped_stones << offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_contiguous_stones_left(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    return s


def search_contiguous_stones_right(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    return s


def bit_count(bit):
    return bin(bit).count('1')


def bit_to_array(bit, size):
    return np.array(list(reversed((("0" * size) + bin(bit)[2:])[-size:])), dtype=np.uint8)
