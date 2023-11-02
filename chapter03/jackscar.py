import matplotlib.pyplot as plt
import numpy as np

class CarLot:
    def __init__(self, start = 20):
        self.max_car_per_lot = 20
        self.lot_a = start
        self.lot_b = start

    def get_state(self):
        return self.lot_a, self.lot_b

    def end_of_day_calc(self):
        pass
        # people return their cars

        # then the cars are maxed out to 20 

        # then people hire their cars

        # then jack takes an action

        # then the next state happens

    def step(self, state, action):
        """MDP dynamics
        Takes in a state and action and returns
        next_state, reward"""
        assert 5 >= action >= -5
        self.lot_a = state[0]
        self.lot_b = state[1]
        if action == 0:
            return self.get_state(), (self.lot_a  * 10) + (self.lot_b * 10)
        elif action > 0:
            max_cars = min(self.lot_a, action)
            self.lot_a = self.lot_a - max_cars
            self.lot_b = self.lot_b + max_cars
        elif action < 0:
            max_cars = min(self.lot_b, action)
            self.lot_a = self.lot_a + max_cars
            self.lot_b = self.lot_b - max_cars
        self.lot_a = min(self.lot_a, 20)
        self.lot_b = min(self.lot_b, 20)
        return self.get_state(), ((self.lot_a  * 10) + (self.lot_b * 10) ) - (max_cars * 2)



