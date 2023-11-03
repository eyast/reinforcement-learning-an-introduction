# implementation of the Splippery Walk Five MDP from Grokking Deep Reinforcement LEarning book page 127 section 3

import numpy as np

class SlipperyWalk:
    def __init__(self):
        pass
    
    def step(state, action):
        """MDP transition"""
        if state == 0:
            return 0, 0
        elif state == 6:
            return 6, 0
        else:
            next_state = state + action
            if next_state == 6:
                return 6, 1
            else:
                return next_state, 0
            
if __name__ == "__main__":
    discount = 0.99
    v = np.zeros(7)
    while True:
