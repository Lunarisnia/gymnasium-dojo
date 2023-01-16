
# Task: Solving Blackjack with Q-Learning
# Q-Learning learning adalah salah satu algoritma Reinforcement Learning
# yang berjalan menggunakan sebuah Q-Table, Q-Table adalah kumpulan
# state dari hasil observasi yang dijadikan key dan berisi opsi pilihan terbaik untuk kondisi tersebut
# Proses learning pada Q-Learning pada dasarnya hanya proses untuk membuat Q-Table tersebut.

from __future__ import annotations

# For creating a dictionary with custom default response instead of the default KeyError
from collections import defaultdict

# Plotting a data
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# For Easier good looking data plotting
import seaborn as sns

# For Creating shapes easily, also why is this called Artist?
from matplotlib.patches import Patch

# For creating progress bar
from tqdm import tqdm

import gymnasium as gym
# Creating the blackjack environtment.
env = gym.make('Blackjack-v1', sab=True)

# Reset the environtment to get the first observation
done = False
observation, info = env.reset()
print("observation =", observation)
print("info =", info)
# Observation is a Tuple with 3 values (16, 9, False)
# - The players current Sum
# - Value of the dealers face-up card
# - Boolean whether the player holds a usable ace (An Ace is usable if it counts as 11 without busting)

action = env.action_space.sample()
print("action =", action)  # action = 1

# execute the action in our environment and receive infos from the environment.
observation, reward, terminated, truncated, info = env.step(action)
print(observation, reward, terminated, truncated, info)
# observation=(24, 10, False)
# reward=-1.0
# terminated=True
# truncated=False
# info={}


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        intial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a Learning rate and an epsilon.

        Args:
            learning_rate: The Learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-Value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        # q_values = {
        #     (13, 6, False): [0. 0.01],
        #     (20, 16, False): [-0.01 0.],
        # }

        self.lr = learning_rate
        # lr = 0.01
        
        self.discount_factor = discount_factor
        # discount_factor = 0.95

        self.epsilon = intial_epsilon
        # epsilon = 0.99999
        self.epsilon_decay = epsilon_decay
        # epsilon_decay = 1e-5
        self.final_epsilon = final_epsilon
        # final_epsilon = 0.1

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration
        """
        # With probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        # q_values = {
        #   (13, 6, False): [0. 0.01],
        #   (20, 16, False): [-0.01 0.],
        # }

        # lr = 0.01
        # obs = (2, 10, False)
        # next_obs = (12, 11, False)
        # reward = 1.5
        # action = 1
        # terminated = False
        # truncated = False
        # info = {}
        # discount_factor = 0.95
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        # q_values[(12, 11, False)] = [0. 0.] = 0.
        # future_q_value = 1 * 0 = 0

        # Why do we need to subtract the future q-value with the old one?
        temporal_difference = (
            reward + self.discount_factor *
            future_q_value - self.q_values[obs][action]
        )
        # PEMDAS: Parenthesis, Exponential, Multiplication, Addition, Subtraction
        # q_values[(2, 10, False)][0] = 0.0
        # temporal_difference = 1.5 + 0.95 * 0 - 0 = -1

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        # q_values[(20, 16, False)][0] = 0.0
        # q_values[(20, 16, False)][0] = 0.0 + 0.01 * -1.0 = 0.0 - 0.01 = -0.01
        
        self.training_error.append(temporal_difference)
        # training_error = [1.0]
        # append = [1.0, -1]

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)
        # epsilon = max(0.1, (1.0 - 1e-5)) 
        # = max(0.1, 0.99999) = 0.99999


# Agent is done time to train
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
# reduce the exploration over time
epsilon_decay = start_epsilon / (n_episodes / 2)
# epsilon_decay = 1e-5
final_epsilon = 0.1
agent = BlackjackAgent(learning_rate=learning_rate,
                       intial_epsilon=start_epsilon,
                       epsilon_decay=epsilon_decay,
                       final_epsilon=final_epsilon)
 
# Training

"""
========
EPISODE 1
========


1st Iteration

q_values = {
   (13, 6, False): [0. 0.01]
}
training_error = [1.0]

lr = 0.01
obs = (13, 6, False)
next_obs = (20, 16, False)
reward = 1.0
action = 1
terminated = False
truncated = False
info = {}
discount_factor = 0.95

future_q_value = 1 * 0 = 0
temporal_difference = 1 + 0.95 * 0 - 0 = 1.0
q_values[(13, 6, False)][1] = 0.0 + 0.01 * 1.0 = 0.0 + 0.01 = 0.01
training_error = [] = [1.0]


2nd Iteration
q_values = {
  (13, 6, False): [0. 0.01],
  (20, 16, False): [-0.01 0.],
}
training_error = [1.0, -1]

lr = 0.01
obs = (20, 16, False)
next_obs = (29, 20, False)
reward = -1.0
action = 0
terminated = True
truncated = False
info = {}
discount_factor = 0.95

future_q_value = 0 * 0 = 0
temporal_difference = -1 + 0.95 * 0 - 0 = -1


epsilon = 0.99999



=======
EPISODE 2
=======
"""

# Todo: Create an interactive menu to view value during training

# Record cumulative rewards and episode lengths
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    # obs = (2, 10, False)
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        # 1st iteration: Random sample = 1

        next_obs, reward, terminated, truncated, info = env.step(action)
        

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    # Why is it only updated once the episode is over?
    # Maybe because we want to try a single epsilon for an episode to see how effective that is
    agent.decay_epsilon()

# Todo: Learn how the code actually works
# Todo: Visualize the training