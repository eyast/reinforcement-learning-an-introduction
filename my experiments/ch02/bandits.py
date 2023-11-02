import numpy as np
import matplotlib.pyplot as plt
import time


class Bandit:
    """Defines a multi-arm bandit where rewards are stationary
    
    Takes in as parameters:
    - list_of_means: a list of the mu value for each arm
    """
    def __init__(self, k):
        assert isinstance(k, int)
        self.k = k
        self.means = np.random.normal(0, 1, k)

    def draw(self, k: int) -> float:
        """Draws am arm of the bandit, returns a reward"""
        return np.random.normal(self.means[k], 1)
    
    def get_max_reward(self):
        """return the maximum attainable reward"""
        return np.max(self.means)


class Agent:
    """an Agent that plays a multi-armed bandit.
    it can query how many k arms the Bandit has.
    and it works at solving this bandit using different algorithms.
    
    functions:
    - get_k(): gets the number of k arms the bandit has (action space size)
    - choose(): decide which arm to pull. Returns the arm (int)

    constructor parameters:
    - k = number of arms bandit has
    - n = number of maximum trial for each episode
    - mode = "greedy", e-greedy, UCB
    - for e-greedy:
        eps = epsilon value
    - for UCB:
        determine the constant value C
    """
    def __init__(self, bandit: Bandit, episode_length: int, initial_reward = 0, **kwargs):
        self.bandit = bandit
        self.k = bandit.k
        self.episode_length = episode_length
        self.config = kwargs
        self.validate_config()
        self.log = np.zeros((k, episode_length))
        self.t = 0

    def validate_config(self):
        """Validates that the configuration provided to the function is complete
        for the functions to run."""
        assert self.config["mode"], "Provide a mode [greedy, e-greedy, ucb]"
        if self.config["mode"] == "e-greedy":
            assert self.config["eps"], "Provide an epsilon value"
        if self.config["mode"] == "ucb":
            assert self.config["c"], "Provide an value for UCB c constant."

    def choose(self):
        """Choose which arm to pull"""
        if self.config["mode"] == "greedy":
            arm = self.greedy()
        elif self.config["mode"] == "e-greedy":
            arm = self.e_greedy()
        elif self.config["mode"] == "ucb":
            arm = self.ucb()
        reward = self.bandit.draw(arm)
        self.update_action_value(arm, reward, self.t)

    def average_reward(self):
        """Returns the average reward"""
        accumulated_rewards = np.sum(self.log) / self.t
        return accumulated_rewards 

    def is_over(self):
        return self.t == self.episode_length
    
    def e_greedy(self):
        rand = np.random.random()
        if rand < self.config["eps"]:
            return np.random.randint(0, self.k)
        else:
            return self.greedy()

    def greedy(self):
        if self.t == 0:
            return np.random.randint(0, self.k )
        else:
            sums = np.sum(self.log, axis = 1)
            nums = np.where(self.log != 0, True, False)
            nums = np.sum(nums, axis = 1)
            average = sums / nums
            best_arm = np.argmax(np.nan_to_num(average))
            return best_arm

    def update_action_value(self, arm, reward, t):
        self.log[arm, t] = reward
        self.t += 1

if __name__ == "__main__":
    k = 10
    episode_length = 2000
    num_episodes = 15
    r_1, r_2, r_3 = [], [], []
    for i in range(num_episodes):
        bandit = Bandit(k)
        agent_1 = Agent(bandit, episode_length, mode="greedy")
        agent_2 = Agent(bandit, episode_length, mode="e-greedy", eps=0.1)
        agent_3 = Agent(bandit, episode_length, mode="e-greedy", eps=0.1)
        while not agent_1.is_over():
            agent_1.choose()
        r_1.append(agent_1.average_reward())
        while not agent_2.is_over():
            agent_2.choose()
        r_2.append(agent_2.average_reward())
        while not agent_3.is_over():
            agent_3.choose()
        r_3.append(agent_1.average_reward())
    rolling_average = np.zeros((3, num_episodes))
    rolling_average[0, 0] = r_1[0]
    rolling_average[0, 1] = r_2[0]
    rolling_average[0, 2] = r_3[0]
    for idx, r in enumerate([r_1, r_2, r_3]):
        for i in range(1, num_episodes):
            rolling_average[idx, i] = ((rolling_average[idx, i - 1] * (i -  1)) + r[i]) / i
    labels = ["greedy", "eps 0.1", "eps 0.01"]
    for i in range(3):
        plt.plot(rolling_average[i,:], label=labels[i], alpha=0.4)
    plt.legend()
    plt.show()