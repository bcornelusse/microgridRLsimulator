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


class MicrogridEnv(gym.Env):

    def __init__(self, start_date='20150101T00:00:00', end_date='20150102T00:00:00', data_file='case1',
                 decision_horizon=1, results_folder=None, results_file=None):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        :param decision_horizon:
        :param results_folder: if None, set to default location
        :param results_file: if None, set to default file
        """

        self.simulator = Simulator(start_date,
                                   end_date,
                                   data_file,
                                   decision_horizon=decision_horizon,
                                   results_folder=results_folder,
                                   results_file=results_file)
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
            self.state = self.state_refactoring(self.simulator.reset())
        else:
            self.state = state
        return np.array(self.state)

    def step(self, action, state=None):
        """
        Step function, as in gym.
        May also accept a state as input (useful for MCTS, for instance).
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if state is None:
            state = self.state
        state_formatted = self.state_formatting(state)
        next_state, reward, done = self.simulator.step(state_formatted, action)
        self.state = self.state_refactoring(next_state)

        return np.array(self.state), reward, done, {}

    def state_refactoring(self, state):
        """
        Convenience function that flattens the received state into an array

        :param state: State of the agent as a list
        :return: Flattened representation of the state as an array
        """
        state_array = np.concatenate((np.array([state[0]]), np.array(state[1]).reshape(-1), np.array([state[2]])),
                                     axis=0)

        return state_array

    def state_formatting(self, state_array):
        """
        Inverse of state_refactoring
        """
        n = len(self.simulator.grid.storages)
        state = [state_array[0],  state_array[1:1+n].tolist(),  state_array[-1]]

        return state
