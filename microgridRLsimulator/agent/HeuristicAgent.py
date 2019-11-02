# -*- coding: utf-8 -*-

from microgridRLsimulator.agent.agent import Agent
from microgridRLsimulator.utils import positive, negative

import numpy as np
from copy import deepcopy
from microgridRLsimulator.utils import time_string_for_storing_results


class HeuristicAgent(Agent):

    def __init__(self, env, n_test_episodes=1):
        super().__init__(env)
        self.n_test_episodes = n_test_episodes
        #self.num_actions = self.env.action_space.n
        #self.state_size = env.observation_space.shape[0] #TODO I am not sure whether it is useful somewhere

    @staticmethod
    def name():
        return "Heuristic"

    def train_agent(self):
        pass

    def simulate_agent(self, agent_options=None):
        for i in range(1, self.n_test_episodes + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                # Charge if production > demand, discharge if production < demand, Idle otherwise
                consumption = self.env.simulator.grid_states[-1].non_steerable_consumption
                nb_storage = len(self.env.simulator.grid_states[-1].state_of_charge)
                production = self.env.simulator.grid_states[-1].non_steerable_production
                
                soc = self.env.simulator.grid_states[-1].state_of_charge
                total_possible_discharge = 0
                for b in range(len(soc)):
                    storage = self.env.simulator.grid.storages[b]
                    total_possible_discharge += min(soc[b] * storage.discharge_efficiency / self.env.simulator.grid.period_duration,
                                                storage.max_discharge_rate)
                net_gen = production - consumption 
                if positive(net_gen):
                    action = self.env.simulator.high_level_actions.index(
                        tuple('C' for x in range(nb_storage)))  # Charge action for all storages
                elif negative(net_gen):
                    if total_possible_discharge + production > consumption:
                        action = self.env.simulator.high_level_actions.index(tuple('D' for x in range(nb_storage)))
                    else:
                        action = self.env.simulator.high_level_actions.index(
                        tuple('C' for x in range(nb_storage)))
                else:
                    action = self.env.simulator.high_level_actions.index(tuple('I' for x in range(nb_storage)))
                next_state, reward, done, info = self.env.step(state=state, action=action)
                # reward = self.reward_function(reward_info)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation: %d and the reward is: %d.' % (i, cumulative_reward))
            self.env.simulator.store_and_plot(
                folder="results/" + self.name() + "/" + self.env.simulator.case + "/" + time_string_for_storing_results(
                    self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime(
                        "%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                    self.env.simulator.case) + "_" + str(i), agent_options=agent_options)

    def reward_function(self, reward_info):
        """
        Method that transforms the reward infos into a reward value with the help of a reward function tuned by the user.

        :param reward_info: dictionary that contains reward information relative to the chosen objectives 
        (total_cost, fuel_cost, load_shedding, curtailment, storage_maintenance).
        :return: reward value from a tuned reward function.
        """
        reward = - reward_info["total_cost"]
        return reward

    def set_environment(self, env):
        super().set_environment(env)


agent_type = HeuristicAgent
