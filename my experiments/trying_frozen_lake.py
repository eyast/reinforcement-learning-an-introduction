import gymnasium as gym
import time

env = gym.make('FrozenLake-v1', render_mode="human", map_name="4x4")
env.reset()
done = False

while not done:
    #env.render()
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    time.sleep(1)