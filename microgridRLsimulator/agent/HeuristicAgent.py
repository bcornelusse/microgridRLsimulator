# -*- coding: utf-8 -*-

from microgridRLsimulator.agent.agent import Agent
from microgridRLsimulator.utils import positive, negative

import numpy as np
from copy import deepcopy


class HeuristicAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
    
    @staticmethod
    def name():
        return "Heuristic"

    def train_agent(self):
        pass #Nothing to train the Heuristic agent with

    def simulate_agent(self, simulation_steps=1):
        for i in range(1, simulation_steps + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                state_array = self.state_refactoring(state)
                # Charge if production > demand, discharge if production < demand, Idle otherwise
                consumption = state[0]
                production = state[2]
                nb_storage = len(state[1])
                net_gen = production - consumption
                if positive(net_gen):
                    action = self.env.simulator.high_level_actions.index(tuple('C' for x in range(nb_storage))) # Charge action for all storages
                elif negative(net_gen):
                    action = self.env.simulator.high_level_actions.index(tuple('D' for x in range(nb_storage)))
                else:
                    action = self.env.simulator.high_level_actions.index(tuple('I' for x in range(nb_storage)))
                next_state, reward, done, info = self.env.step(state = state, action= action)
                # reward = self.reward_function(reward_info)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation - the reward is: %d.' % (cumulative_reward))
        self.env.simulator.store_and_plot()
    
    def reward_function(self, reward_info):
        """
        Method that transforms the reward infos into a reward value with the help of a reward function tuned by the user.

        :param reward_info: dictionary that contains reward information relative to the chosen objectives 
        (total_cost, fuel_cost, load_shedding, curtailment, storage_maintenance).
        :return: reward value from a tuned reward function.
        """
        reward = - reward_info["total_cost"]
        return reward


agent_type = HeuristicAgent