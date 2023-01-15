
# Task: Solving Blackjack with Q-Learning

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
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

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = intial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

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
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor *
            future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)


# Agent is done time to train
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
# reduce the exploration over time
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(learning_rate=learning_rate,
                       intial_epsilon=start_epsilon,
                       epsilon_decay=epsilon_decay,
                       final_epsilon=final_epsilon)
 
# Training
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

# Todo: Learn how the code actually works
# Todo: Visualize the training