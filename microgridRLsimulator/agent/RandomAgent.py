# -*- coding: utf-8 -*-

from microgridRLsimulator.agent.agent import Agent

import numpy as np
from copy import deepcopy


class RandomAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
    
    @staticmethod
    def name():
        return "Random"

    def train_agent(self):
        pass #Nothing to train the Random agent with

    def simulate_agent(self, simulation_steps=1):
        for i in range(1, simulation_steps + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                state_array = self.state_refactoring(state)
                # Take a random action from the action space
                action = np.random.choice(len(self.env.high_level_actions))
                next_state, reward, done = self.env.step(state, action)
                #reward = self.reward_function(reward_info)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation: %d and the reward is: %d.' % (i, cumulative_reward))
        self.env.store_and_plot()

    def reward_function(self, reward_info):
        """
        Method that transforms the reward infos into a reward value with the help of a reward function tuned by the user.

        :param reward_info: dictionary that contains reward information relative to the chosen objectives 
        (total_cost, fuel_cost, load_shedding, curtailment, storage_maintenance).
        :return: reward value from a tuned reward function.
        """
        reward = - reward_info["total_cost"]
        return reward

agent_type = RandomAgent
