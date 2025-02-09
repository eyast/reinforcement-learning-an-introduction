import gymnasium as gym
from mces_agent import MCESAgent

EPISODE_LENGTH = 100
NUM_EPISODES = 1000
GAMMA = 0.9
NUM_POLICY_ITERATIONS = 1000

env = gym.make('FrozenLake-v1',
               map_name="8x8",
               random_start=True,
               is_slippery=False)

agent = MCESAgent(env=env, 
                  gamma=GAMMA, 
                  max_episode_length=EPISODE_LENGTH, 
                  total_trajectories=NUM_EPISODES,
                  num_policy_iterations=NUM_POLICY_ITERATIONS)

agent.iterate_x_policies()