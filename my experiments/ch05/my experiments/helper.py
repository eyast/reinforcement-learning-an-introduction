import numpy as np
import matplotlib.pyplot as plt


action_text_map = {
    0: "left",
    1: "down",
    2: "right",
    3: "up"
}

def convert_1d_to_2d_array(array, env):
    """Takes the 1D array and converts it to numpy for troubleshooting"""
    obs_space = env.observation_space.n
    side_length = int(np.sqrt(obs_space))
    return array.reshape((side_length , side_length ))


def generate_randompolicy(env):
    """Generates a random policy based on an environment"""
    obs_space = env.observation_space.n
    action_space = env.action_space.n
    return np.random.randint(0, action_space, size=obs_space)



def display_policy_ax(ax, policy):
    """
    Displays the FrozenLake policy on the given axis.
    Each cell is annotated with an arrow representing the action.
    
    Args:
        ax: Matplotlib axis object.
        policy: 2D numpy array of actions (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP).
    """
    grid_shape = policy.shape
    arrow_mapping = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    
    # Configure grid lines.
    ax.set_xticks(np.arange(-0.5, grid_shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Annotate each cell with the corresponding arrow.
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            action = policy[i, j]
            arrow = arrow_mapping.get(action, '')
            ax.text(j, i, arrow, ha='center', va='center', fontsize=24)
    
    # Adjust the axis limits so (0,0) is top-left.
    ax.set_xlim(-0.5, grid_shape[1]-0.5)
    ax.set_ylim(grid_shape[0]-0.5, -0.5)


def convert_obs_to_array_location(obs):
    row = obs // 8
    col = obs % 8
    return row, col


def compute_returns(trajectory, gamma):
    rewards = [x[2] for x in trajectory]
    returns = [0] * len(rewards)
    for i, idx in enumerate(returns):
        if i == 0:
            returns[0] = gamma
        else:
            returns[i] = returns[i-1] * gamma
    returns = returns[::-1]
    for i, _ in enumerate(trajectory):
        trajectory[i][2] = returns[i]
    return trajectory

def display_val_policy(value_function, np_policy, success_rate, agent_id):
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
    plt.title(f'Agent ID: {agent_id}, Success Rate: {success_rate:.2f}')
    plt.show()


def plot_last_three_iterations(self, values, policies, episode, success):
    """Plots the value functions and policies of the last three iterations"""
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        np_val = convert_1d_to_2d_array(values[i], self.env)
        np_pol = convert_1d_to_2d_array(policies[i], self.env)
        axs[i, 0].imshow(np_val, cmap='viridis')
        axs[i, 0].set_title(f'Value Function Iteration {episode + i} - {success}')
        # axs[i, 1].imshow(np_pol, cmap='viridis')
        display_policy_ax(axs[i, 1], np_pol)
        axs[i, 1].set_title(f'Policy Iteration ')
        if i > 0:
            prev_pol = convert_1d_to_2d_array(policies[i-1], self.env)
            for row in range(np_pol.shape[0]):
                for col in range(np_pol.shape[1]):
                    if np_pol[row, col] != prev_pol[row, col]:
                        axs[i, 1].text(col, row, 'X', ha='center', va='center', color='red', fontsize=24)
    plt.tight_layout()
    plt.show()