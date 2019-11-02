# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, env):
        self.env = env

    def run(self):
        """
        Method that is launched in order to train and simulate the agent
        :return: Nothing
        """
        self.train_agent()
        self.simulate_agent()

    @abstractmethod
    def train_agent(self):
        """
        Method used to train the RL agent
        :return: Nothing
        """
        pass

    @abstractmethod
    def reset_agent(self):
        """
        Method used to reset the informatino of  the RL agent
        :return: Nothing
        """
        pass

    @abstractmethod
    def simulate_agent(self, simulation_steps=1, agent_options=None):
        """
        Method that is simulating the result of the training process
        :param simulation_steps: Number of times the simulation is repeated (> 1 if stochastic environment).
        """
        pass

    # @staticmethod
    # def state_refactoring(state):
    #     """
    #     Convenience function that flattens the received state into an array

    #     :param state: State of the agent as a list
    #     :return: Flattened representation of the state as an array
    #     """
    #     state_array = np.concatenate((np.array([state[0]]), np.array(state[1]).reshape(-1), np.array([state[2]]), np.array([state[3]])),
    #                                  axis=0)

    #     return state_array

    def set_environment(self, env):
        self.env = env
