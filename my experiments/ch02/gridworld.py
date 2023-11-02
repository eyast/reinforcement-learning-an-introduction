import numpy as np

actions = [0, 1, 2, 3] # (representing left, up, down, right respectively)

class Gridworld:
    def __init__(self, size):
        self.size = size
        self.v = np.zeros((size, size))

    def get_probs(self, state, action):
        """Given a state and an action, returns:

        input: state: tuple
        action: int

        - next state
        - reward"""
        if state == (0, 1):
            return (4, 1), 10
        if state == (0, 3):
            return (2, 3), 5
    # left
        if state[1] == 0 and action == 0:
            return state, -1
    # right
        if state[1] == self.size - 1 and action == 3:
            return state, -1
    # top
        if state[0] == 0 and action == 1:
            return state, -1
    # down
        if state[0] == self.size - 1 and action == 2:
            return state, -1

        else:
            if action == 0:
                return (state[0], state[1]  - 1), 0
            if action == 3:
                return (state[0], state[1]  + 1), 0
            if action == 1:
                return (state[0] - 1, state[1]), 0
            if action == 2:
                return (state[0] + 1, state[1]), 0

eps = 1e-6 
gamma = 0.9
grid_world = Gridworld(5)
v = np.zeros((5, 5))
done = False
while not done:
    v_copy = np.zeros_like(v)
    for idx_row, _ in enumerate(v):
        for idx_col, _ in enumerate(_):
            state = (idx_row, idx_col)
            for action in actions:
                next_state, reward = grid_world.get_probs(state, action)
                v_copy[state[0], state[1]] += 0.25 * (reward + gamma * v[next_state[0], next_state[1]])
    diff = np.sum(np.abs(v - v_copy))
    if diff < eps:
        done = True
    else:
        v = v_copy

print(np.round(v, 1))