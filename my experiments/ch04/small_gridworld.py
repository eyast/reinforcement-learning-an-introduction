import numpy as np

ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]

class SmallGridWorld:
    def __init__(self):
        self.board =np.zeros((4, 4))
        self.board[0, 0] = 1
        self.board[3, 3] = 1

    def step(self, s, action):
        # terminal states
        if s == (0, 0) or s == (3, 3):
            return s, 0
        # edge cases:
        if s[0] == 0 and action == 