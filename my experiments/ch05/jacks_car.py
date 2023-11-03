# Definition of the MDP of RLAI Chapter 5 page 81
from scipy.stats import poisson
import numpy as np
import time

CARS = 21
lam_rental_a = 3

class PoissonProbabilities:
    """A class that constructs the probabilities and stores them"""
    def __init__(self):
        self.probs = np.zeros((2, 2, 21)) # (lot_a, Lot_b) * (rental, return) * up_to_21_cars
        lam_rental_a = poisson.pmf(range(21), 3)
        lam_rental_b = poisson.pmf(range(21), 4)
        lam_return_a = poisson.pmf(range(21), 3)
        lam_return_b = poisson.pmf(range(21), 2)
        self.probs[0, 0, :] = lam_rental_a
        self.probs[0, 1, :] = lam_return_a
        self.probs[1, 0, :] = lam_rental_b
        self.probs[1, 1, :] = lam_return_b

    def get_prob(self, lot, type, quantity):
        """Returns the probability that an event happens at a lot based on a quantity"""
        if type == "return":
            type = 1
        elif type == "rental":
            type = 0
        return self.probs[lot, type, quantity]

class JacksEnv:
    """Defines an MDP for Jack's environment"""
    def __init__(self):
        self.actions = list(range(-5, 5+1, 1))

    def transition(self, state, action):
        returns = 0
        returns -= 2 * abs(action)
        