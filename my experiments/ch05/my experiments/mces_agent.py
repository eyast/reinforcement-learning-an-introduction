# Defines the Monte Carlo Exploration starting agent
# and its helping functions
import numpy as np
from helper import (
    generate_randompolicy, 
    compute_returns, 
    display_val_policy, 
    convert_1d_to_2d_array,
    convert_obs_to_array_location,
    plot_last_three_iterations
    )

class MCESAgent:
    def __init__(self, env=None, 
                gamma=None,
                max_episode_length=None, 
                total_trajectories = None,
                num_policy_iterations=None, 
                parent=None,
                id=0, 
                policy=None):
        """A Monte Carlo Exploring Start Agent
        
        env: A Gymnasium environment
        Gamma: Discount factor
        max_episode_length: the maximum episode_length
        total_trajectories: total number of trajectories to sample
        num_policy_iterations: the number of times to perform Geneal Policy Iteration
        
        """
        assert env, "Environment must be provided"
        assert gamma, "Gamma value must be provided"
        assert max_episode_length, "Maximum episode length must be set"
        assert total_trajectories, "You must set the number of total trajectories to sample"
        assert num_policy_iterations, "You must define how many times to perform Policy Iterations"

        self.parent = parent
        self.env = env
        self.gamma = gamma
        self.max_episode_length = max_episode_length
        self.total_trajectories = total_trajectories
        self.id = id
        self.episodes_played = 0
        self.steps_taken = 0
        self.total_reward = 0
        self.successful_envs = []
        self.num_policy_iterations = num_policy_iterations

        if policy is None:
            self.policy = generate_randompolicy(env) if policy is None else self.policy
        else:
            self.policy = policy
        # self.value_function = np.zeros_like(self.policy, dtype=np.float32)
        # self.count_function = np.zeros_like(self.policy)
        if parent:
            self.action_function = self.parent.action_function
            self.count_function = self.parent.count_function
        else:
            self.action_function = np.zeros ((4, 64), dtype=np.float32)
            self.count_function = np.zeros ((4, 64), dtype=np.float32)

    def take_action(self, obs):
        return self.policy[obs]

    def play_single_episode(self, episode_length, render=False):
        """Plays one episode completely and update the value function"""
        self.episodes_played += 1
        env = self.env
        prev_obs = env.reset()
        prev_obs = prev_obs[0]
        step = 0
        done = False
        trajectory = []
        while not done and step < episode_length:
            if render:
                env.render()
            action = self.take_action(prev_obs)
            obs, reward, done, info, _ = env.step(action)
            trajectory.append([prev_obs, action, reward])
            prev_obs = obs
            step += 1
        self.steps_taken += step
        if reward != 0:
            trajectory = compute_returns(trajectory, self.gamma)
            self.update_value_function(trajectory)
            self.total_reward += reward

    def update_value_function(self, trajectory):
        for i, node in enumerate(trajectory):
            obs, action, reward = node
            if self.count_function[action, obs] == 0:
                # self.value_function[obs] = reward
                self.action_function[action, obs] = reward
                self.count_function[action, obs] = 1
            else:
                self.action_function[action, obs] =  (self.action_function[action, obs] * np.sum(self.count_function[:,obs]) + reward) / ( np.sum(self.count_function[:,obs]) + 1)
                # self.value_function[obs] = (self.value_function[obs] * (self.count_function[obs]) + reward) / (self.count_function[obs] + 1)
                self.count_function[action, obs] += 1


    def play(self, debug=False):
        """Plays using a single policy"""
        for i in range(self.total_trajectories):
            self.play_single_episode(self.max_episode_length)
        print(f"playing {self.total_trajectories} episodes of length {self.max_episode_length} each.")
        success_rate = self.total_reward/self.total_trajectories
        self.value_function = np.max(self.action_function, axis=0)
        if debug:
            np_val = convert_1d_to_2d_array(self.value_function, self.env)
            np_pol = convert_1d_to_2d_array(self.policy, self.env)
            display_val_policy(np_val, np_pol, success_rate, self.id)
        return success_rate


    def generate_new_policy(self):
        """Generates a new policy"""
        new_policy = self.policy_improvement()
        return MCESAgent(env=self.env,
                        gamma=self.gamma,
                        max_episode_length=self.max_episode_length,
                        total_trajectories=self.total_trajectories,
                        num_policy_iterations=self.num_policy_iterations,
                        parent=self,
                        id=self.id + 1, 
                        policy=new_policy)
    
    def iterate_x_policies(self):
        """Iterates through policies and updates the agent"""
        agent = self
        last_three_values = []
        last_three_policies = []
        for i in range(self.num_policy_iterations):
            success = agent.play(debug=False)
            last_three_values.append(agent.value_function)
            last_three_policies.append(convert_1d_to_2d_array(agent.policy, self.env))
            print("Generating a new policy")
            agent = agent.generate_new_policy()
            if len(last_three_policies) > 2 and i % 10 == 0:
                plot_last_three_iterations(self, last_three_values, last_three_policies, i, success)
                last_three_values = []
                last_three_policies = []
        return agent

    def policy_improvement(self):
        """Silly policy improvement based on Argmax the value function. Does not explore the complete
        State-action pairs"""
        new_policy = np.zeros_like(self.policy)
        num_states = self.action_function.shape[1]
        for state in range(num_states):
            values = self.action_function[:,state]
            if np.sum(values)> 0:
                new_policy[state] = np.argmax(values) if np.random.uniform(0, 1) < 0.95 else np.random.randint(0, 4)
            else:
                new_policy[state] = np.random.randint(0, 4)
        return new_policy

        