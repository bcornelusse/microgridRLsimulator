"""
This demo shows how to create an interact with an environment.
"""

import numpy as np
from microgridRLsimulator.gym_wrapper import MicrogridEnv


# Initialize environment
env = MicrogridEnv()

# Optional args:
# env = MicrogridEnv(start_date='20150101T00:00:00', end_date='20150102T00:00:00', data_file='case1',
#                    decision_horizon=1, results_folder=None, results_file=None)

# Compute cumulative reward of a random policy
sum_reward = 0
T = 24  # horizon
for tt in range(T):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    sum_reward += reward

# Store and plot
env.simulator.store_and_plot()
