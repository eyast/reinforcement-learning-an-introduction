### Monte Carlo Exploration Start policy evaluation and improvement
# Gym environment modified to start in random states
# does not explore every single state-action pair, but instead starts in any random state and starts with a random policy

# Todo - Explore every state action pair
# Visual the resulted policy
# Find (plan) the best Value function and action value based on Bellman
#Todo - generalize helper/convert_obs_to_array_location for any environment
#Todo - Generalize display policy to other text toy examples



import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from helper import display_policy_ax, convert_obs_to_array_location, compute_returns



action_text_map = {
    0: "left",
    1: "down",
    2: "right",
    3: "up"
}






# env = gym.make('FrozenLake-v1', render_mode="human", map_name="8x8", random_start=True)
env = gym.make('FrozenLake-v1', map_name="8x8", random_start=True)
policy = np.random.randint(0, 4, size=64)
np_policy = np.zeros((8, 8))
for idx, i in enumerate(policy):
    row, col = convert_obs_to_array_location(idx)
    np_policy[row, col] = i


def play_episode(policy, env):
    prev_obs = env.reset()
    prev_obs = prev_obs[0]
    step = 0
    done = False
    trajectory = []
    while not done:  
        env.render()
        action = policy[prev_obs]
        obs, reward, done, info, _ = env.step(action)
        trajectory.append([prev_obs, action, reward])
        prev_obs = obs
        
        # time.sleep(1)
        step += 1
    if reward:
        return reward, compute_returns(trajectory, gamma=GAMMA)
    else:
        return reward, trajectory



def update_value_function(episode_data, value_function=value_function):
    for i, node in enumerate(episode_data):
        obs, action, reward = node
        if count_function[0, obs] == 0:
            value_function[0, obs] = reward
            count_function[0, obs] = 1
        else:
            value_function[0, obs] = (value_function[0, obs] * (count_function[0, obs]) + reward) / (count_function[0, obs] + 1)
            count_function[0, obs] += 1
    return value_function


for episode in range(NUM_EPISODES):
    reward, episode_data = play_episode(policy, env)
    if reward:
        value_function = update_value_function(episode_data)

value_function = np.array(value_function[0]).reshape((8,8))


# Create a figure with two subplots.
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: Value Function heatmap.
im = axs[0].imshow(value_function, cmap='viridis', interpolation='nearest')
axs[0].set_title('Value Function')
axs[0].set_xticks([])
axs[0].set_yticks([])
fig.colorbar(im, ax=axs[0])

# Right subplot: Policy visualization using arrows.
display_policy_ax(axs[1], np_policy)
axs[1].set_title('Policy')
print(np_policy)
plt.show()