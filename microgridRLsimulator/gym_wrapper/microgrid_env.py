"""
This file defines a class MicrogridEnv that wraps the Simulator in this package, so that it follows the
OpenAI gym (https://github.com/openai/gym) format.

TODO:
    * verify observation_space
"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from microgridRLsimulator.simulate import Simulator
from datetime import timedelta
import pandas as pd

class MicrogridEnv(gym.Env):

    def __init__(self, start_date, end_date, data_file):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        :param results_folder: if None, set to default location
        :param results_file: if None, set to default file
        """

        self.simulator = Simulator(start_date,
                                   end_date,
                                   data_file)
        self.state = None
        self.action_space = spaces.Discrete(len(self.simulator.high_level_actions))

        # Observation space
        high = 1e3*np.ones(2 + len(self.simulator.grid.storages))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = self.simulator.reset()
        else:
            self.state = state
        return np.array(self.state_refactoring(self.state))

    def step(self, action, state=None):
        """
        Step function, as in gym.
        May also accept a state as input (useful for MCTS, for instance).
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if state is None:
            state = self.state
        state_formatted = self.state_formatting(state)
        next_state, reward, done = self.simulator.step(high_level_action = action, state = state_formatted)
        self.state = self.state_refactoring(next_state)

        return np.array(self.state), reward, done, {}

    def state_refactoring(self, state):
        """
        Convenience function that flattens the received state into an array

        :param state: State of the agent as a list
        :return: Flattened representation of the state as an array
        """
        consumption = state[0]
        storages_soc = state[1]
        production = state[2]
        delta_t = state[3]
        state_array = np.concatenate((np.array([consumption]), np.array(storages_soc).reshape(-1), np.array([production]), np.array([delta_t])),
                                     axis=0)
        return state_array

    def state_formatting(self, state_array):
        """
        Inverse of state_refactoring
        """
        n= len(self.simulator.grid.storages)
        consumption = state_array[0]
        storages_soc = list(state_array[1:1+n])
        production = state_array[n + 1]
        delta_t = state_array[n + 2]
        state = [consumption, storages_soc, production, delta_t]
        return state
