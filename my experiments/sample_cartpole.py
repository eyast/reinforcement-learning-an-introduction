import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="rgb_array")
import time

observation, info = env.reset(seed=42)
for _ in range(1000):
    env.render()
    time.sleep(1)
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()