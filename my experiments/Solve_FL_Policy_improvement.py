# an attempt at solving FrozenLake using Policy Improvement

import gymnasium as gym
import time
import numpy as np

env = gym.make('FrozenLake-v1', render_mode="human", map_name="8x8")
#env = gym.make('FrozenLake-v1', map_name="4x4")
env.reset()

tol = 1e-5
discount = 0.99

# Value function where a random policy is followed (all 4 actions have same weight)
v = np.zeros(64)
while True:
    v_copy = np.zeros(64)
    for state in env.P.keys():
        state_info = env.P[state]
        for action in state_info.keys():
            for prob, next_state, reward, done in state_info[action]:
                v_copy[state] += (1/len(state_info.keys())) * prob * ( reward + discount * v[next_state])
    if np.sum(np.abs(v_copy - v)) < tol:
        break
    v = v_copy
print(v)

# Policy Improvement

# 0: Left
# 1: Down
# 2: Right
# 3: Up

policy = [2] * 64

def policy_eval(policy, env):
    """Takes in a policy and an environment, 
    and returns the value function of following that policy, and how many iterations"""
    v = np.zeros(64)
    iters = 0
    while True:
        v_copy = np.zeros(64)
        for state in env.P.keys():
            state_info = env.P[state]
            for action in state_info.keys():
                if action == policy[state]:
                    for prob, next_state, reward, done in state_info[action]:
                        v_copy[state] +=  prob * ( reward + discount * v[next_state])
        if np.sum(np.abs(v_copy - v)) < tol:
            break
        v = v_copy
        iters += 1
    return v, iters


def policy_improvement(policy, value):
    """Takes in a policy and value function, and if possible, return a better policy
    (one in which the value statement for any state is higher)"""
    policy_stable = True
    for state in env.P.keys():
        old_action = policy[state]
        action_values = np.zeros(4)
        for action in range(4):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + discount * value[next_state])
        best_action = np.argmax(action_values)
        policy[state] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

policy_stable = False
num_iters = 0
while not policy_stable:
    v, iters = policy_eval(policy, env)
    policy, policy_stable = policy_improvement(policy, v)
    num_iters += 1

print(f"Total number of iterations: {num_iters}")
policy_n = np.array(policy).reshape((8, 8))
print(policy_n)
print(v.reshape((8, 8)) )

state = 0
done = False
while not done:
    action = policy[state]
    state, rewards, done, truncated, info = env.step(action)

