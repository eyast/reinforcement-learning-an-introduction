import numpy as np

class Agent:
    """An agent that learns to play a discrete, small Gym environment without getting access
    to its MDP. The workflow is:
    
    - The agent generates a num_episodes trajectories according to its e-greedy policy. for each trajectory:
    - It uses the rewards obtained to update the value function.
    - at the end of the tracjetory, it updates its policy by acting greedily on the value function.

    To use the agent, call the play() function.
    """

    def __init__(self, max_steps, gamma, epsilon, num_episodes, env):
        """Parameters:

        - max_steps: int - the maximum number of steps before terminating an episode.
        - gamma: float - the discount factor.
        - epsilon: float [0, 1] - the fixed rate amount of exploration to perform.
        - num_episodes: int - the number of episodes to play. Each episode represents a unique trajectory"""
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.v = np.zeros(self.num_states)
        self.q = np.zeros((self.num_states, self.num_actions))
        self.n = np.zeros((self.num_states, self.num_actions))


    def play(self):
        """Wrapper function that generates k amounts of sequential trajectories.
        k = num_episodes provided during __init__()
        
        """
        for i in range(self.num_episodes):
            trajectory = self.generate_trajectory()
            self.policy_improvement(trajectory)


    def generate_trajectory(self):
        """Generates a trajectory from an environment.
        


        For each state:
        - Determine whether to explore or exploit an action
        """

    def step(self, state):
        """A function that acts at a state.
        It represents q_(s, a) and should ideally converge
        to q*_(s, a)"""