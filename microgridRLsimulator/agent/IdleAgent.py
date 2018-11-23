# -*- coding: utf-8 -*-

from microgridRLsimulator.agent.agent import Agent

import numpy as np
from copy import deepcopy


class IdleAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
    
    @staticmethod
    def name():
        return "Idle"

    def train_agent(self):
        pass #Nothing to train the Idle agent with

    def simulate_agent(self, simulation_steps=1):
        for i in range(1, simulation_steps + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                state_array = self.state_refactoring(state)
                # Take always the last action in the action space - Idle always
                action = len(self.env.high_level_actions)-1
                next_state, reward, done = self.env.step(state, action)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('i am in episode: %d and the reward is: %d.' % (i, cumulative_reward))
        self.env.store_and_plot()

agent_type = IdleAgent
