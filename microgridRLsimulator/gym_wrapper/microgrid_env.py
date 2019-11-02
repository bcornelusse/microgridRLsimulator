"""
This file defines a class MicrogridEnv that wraps the Simulator in this package, so that it follows the
OpenAI gym (https://github.com/openai/gym) format.

"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from microgridRLsimulator.simulate import Simulator
from datetime import timedelta
import pandas as pd


class MicrogridEnv(gym.Env):

    def __init__(self, start_date, end_date, data_file, purpose="Train"):
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

        if self.simulator.data["action_space"] == "Discrete":
            self.action_space = spaces.Discrete(len(self.simulator.high_level_actions))
        else:
            actions_upper_limits = list()
            actions_lower_limits = [0] * (2 * len(self.simulator.grid.storages) +
                                          len([g for g in self.simulator.grid.generators if g.steerable]))

            actions_upper_limits += [b.capacity * b.max_charge_rate / 100. for b in
                                     self.simulator.grid.storages]  # kWh TODO use the period duration (action in kW)
            actions_upper_limits += [b.capacity * b.max_discharge_rate / 100. for b in
                                     self.simulator.grid.storages]  # same
            actions_upper_limits += [g.capacity for g in self.simulator.grid.generators if g.steerable]

            self.action_space = spaces.Box(np.array(actions_lower_limits), np.array(actions_upper_limits),
                                           dtype=np.float64)
            # For now, the action in continuous mode is a GridAction =/= from what is expected from gym framework 
            # (array of actions value as the action space)
            #  TODO: make appropriate changes

        # Observation space
        state_upper_limits = list()
        state_lower_limits = list()
        for attr, val in sorted(self.simulator.data["features"].items()):
            if val:
                if attr == "non_steerable_production" or attr == "res_gen_capacities":
                    state_upper_limits += [sum(g.capacity for g in self.simulator.grid.generators if not g.steerable)]
                    state_lower_limits += [0.]
                elif attr == "non_steerable_consumption":
                    state_upper_limits += [sum(l.capacity for l in self.simulator.grid.loads)]
                    state_lower_limits += [0.]
                elif attr == "n_cycles":
                    state_upper_limits += [np.Inf for b in
                                           self.simulator.grid.storages]  # High number instead of Inf (if you want to be able to sample from it)
                    state_lower_limits += [0 for b in self.simulator.grid.storages]
                elif attr == "delta_h":
                    state_upper_limits += [np.Inf]
                    state_lower_limits += [0]
                elif attr == "state_of_charge" or attr == "capacities":
                    state_upper_limits += [b.capacity for b in self.simulator.grid.storages]
                    state_lower_limits += [0 for b in self.simulator.grid.storages]

        forecast_lower_limits = []
        forecast_upper_limits = []
        if self.simulator.data["forecast_steps"] > 0:
            for i in range(self.simulator.data["forecast_steps"]):
                forecast_upper_limits += [sum(l.capacity for l in self.simulator.grid.loads)]
                forecast_lower_limits += [0.]  
            for i in range(self.simulator.data["forecast_steps"]):
                forecast_upper_limits += [sum(g.capacity for g in self.simulator.grid.generators if not g.steerable)]
                forecast_lower_limits += [0.]

        self.observation_space = spaces.Box(np.array((self.simulator.data["backcast_steps"] + 1) * state_lower_limits + forecast_lower_limits),
                                            np.array((self.simulator.data["backcast_steps"] + 1) * state_upper_limits + forecast_upper_limits),
                                            dtype=np.float64)


        self.np_random = None
        self.purpose = purpose
        self.state_list_copy = []
        self.seed()
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, state=None):
        if state is None:
            self.state = np.array(self.simulator.reset())
            return self.state
        else:
            self.state = state
            return np.array(self.state)

    def step(self, action=None, state=None):
        """
        Step function, as in gym.
        May also accept a state as input (useful for MCTS, for instance).
        """

        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action)) #Will not work for continuous action since action is a GridAction
        if self.simulator.data["action_space"] == "Discrete":
            assert isinstance(action, (int, np.int64)), "A continuous action space is required for this agent."
            next_state, reward, done = self.simulator.step(high_level_action=action)
        else:
            assert not isinstance(action, (int, np.int64)), "A discrete action space is required for this agent."
            next_state, reward, done = self.simulator.step(low_level_action=action)
        self.state = next_state
        return np.array(self.state), reward, done, {}
