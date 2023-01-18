import gymnasium as gym

def printAvailableEnv():
    print(gym.envs.registry.keys())

def main():
    env = gym.make('LunarLander-v2', render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # Agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    
from collections import defaultdict
import numpy as np
# from tqdm import tqdm

foo = defaultdict(lambda: np.zeros(4))
tup = (1, 20, False)