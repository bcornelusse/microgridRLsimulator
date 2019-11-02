# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
import logging

from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from microgridRLsimulator.agent.agent import Agent
import pickle
import numpy as np
from copy import deepcopy

from microgridRLsimulator.simulate.forecaster import Forecaster
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.utils import time_string_for_storing_results, decode_GridState
from microgridRLsimulator.agent.OptimizationAgent import OptimizationAgent

logger = logging.getLogger(__name__)


def plot_training_progress(y_train, y_pred_train, y_test, y_pred_test):
    fig, axes = plt.subplots(len(y_train[0]), 1, sharex=True)
    fig.suptitle('Train')
    for i in range(len(y_train[0])):
        axes[i].plot(y_train[:, i], label="Original")
        axes[i].plot(y_pred_train[:, i], label="Prediction")
        axes[i].legend()

    fig, axes = plt.subplots(len(y_test[0]), 1, sharex=True)
    fig.suptitle('Test')
    for i in range(len(y_test[0])):
        axes[i].plot(y_test[:, i], label="Original")
        axes[i].plot(y_pred_test[:, i], label="Prediction")
        axes[i].legend()
    plt.show()


class SLAgent(Agent):

    def __init__(self, env, control_horizon_data, simulation_horizon, path_to_stored_experience, path_to_store_models,
                 features, test_size, shuffle, use_forecasts, models_dict, expert_iterations, scale_outputs=False, n_test_episodes=1):
        super().__init__(env)
        self.control_horizon = control_horizon_data
        self.simulation_horizon = simulation_horizon
        self.path_to_stored_experience = path_to_stored_experience
        self.path_to_store_models = path_to_store_models + time_string_for_storing_results("experiment",
                                                                                           env.simulator.case)
        if not os.path.isdir(self.path_to_store_models):
            os.makedirs(self.path_to_store_models)
        self.features = features
        self.use_forecasts = use_forecasts
        self.test_size = test_size
        self.shuffle = shuffle
        self.grid = self.env.simulator.grid
        self.forecaster = None
        self.sl_models = None
        self.create_models_from_dict(models_dict)
        self.inputs = None
        self.outputs = None
        self.states = None
        self.forecasts = None
        self.actions = None
        self.expert = OptimizationAgent(self.env, self.control_horizon, self.simulation_horizon)
        self.scale_outputs = scale_outputs
        self.expert_iterations = expert_iterations
        self.n_test_episodes = n_test_episodes

    @staticmethod
    def name():
        return "SL"

    def reset_agent(self):
        self.forecaster = Forecaster(simulator=self.env.simulator, control_horizon=self.control_horizon)
        self.grid = self.env.simulator.grid

    def train_agent(self):
        # Load the datasets
        self.load_data()
        # Prepare the data
        self.process_data()

        ## Supervised learning model to tune.
        for _ in range(self.expert_iterations):
            new_states = []
            expert_actions = []
            for name, model in self.sl_models.items():
                x_train, x_test, y_train, y_test = train_test_split(self.inputs, self.outputs,
                                                                    test_size=self.test_size,
                                                                    shuffle=self.shuffle)
                model.fit(x_train, y_train)

                y_pred_train = model.predict(x_train)
                y_pred_test = model.predict(x_test)

                logger.info("Model: %s Train set error: %d, Test set error: %d" % (
                    name, mean_squared_error(y_train, y_pred_train),
                    mean_squared_error(y_test, y_pred_test)))
                plot_training_progress(y_train, y_pred_train, y_test, y_pred_test)
                model.fit(self.inputs, self.outputs)

                new_states_model, expert_actions_model = self.augment_training_agent(model)
                new_states += new_states_model
                expert_actions += expert_actions_model
            self.add_new_data_in_experience(new_states, expert_actions)

        for name, model in self.sl_models.items():
            model.fit(self.inputs, self.outputs)
            self.store_model(name, model)

    def simulate_agent(self, agent_options=None):
        for name, model in self.sl_models.items():
            actions = []
            for i in range(1, self.n_test_episodes + 1):
                state = self.env.reset()
                self.reset_agent()
                cumulative_reward = 0.0
                done = False
                while not done:
                    state_decoded = decode_GridState(self.env.simulator.grid_states[-1], self.features)

                    if self.forecasts:
                        self.forecaster.exact_forecast(self.env.simulator.env_step)
                        consumption_forecast, pv_forecast = self.forecaster.get_forecast()
                        final_state = np.concatenate((np.array(state_decoded),
                                                      np.array(consumption_forecast), np.array(pv_forecast)))
                    else:
                        final_state = np.array(state_decoded)

                    state_shape = np.shape(final_state)[0]
                    model_output = model.predict(final_state.reshape(-1, state_shape))[0]
                    action = self.list_to_GridAction(list(model_output))
                    actions.append(model_output)
                    next_state, reward, done, info = self.env.step(state=state, action=action)
                    cumulative_reward += reward
                    state = deepcopy(next_state)
                print('Finished %s simulation for model %s and the reward is: %d.' % (self.env.purpose, name,
                                                                                      cumulative_reward))
                self.env.simulator.store_and_plot(
            folder="results/" + time_string_for_storing_results(self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime("%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                                                                self.env.simulator.case) + "_" + str(i), agent_options=agent_options)
            # plt.figure()
            # actions_array = np.array(actions).T
            # for a in actions_array:
            #     plt.plot(a)
            # plt.show()

    def augment_training_agent(self, model):

        state = self.env.reset()
        self.reset_agent()
        self.expert.reset_agent()
        cumulative_reward = 0.0
        done = False
        new_states = []
        expert_actions = []
        while not done:
            self.expert._create_model(self.env.simulator.env_step)
            state_decoded = decode_GridState(self.env.simulator.grid_states[-1], self.features)
            expert_action = self.expert.get_optimal_action()[0].to_list()

            if self.forecasts:
                self.forecaster.exact_forecast(self.env.simulator.env_step)
                consumption_forecast, pv_forecast = self.forecaster.get_forecast()
                final_state = np.concatenate((np.array(state_decoded),
                                              np.array(consumption_forecast), np.array(pv_forecast)))
            else:
                final_state = np.array(state_decoded)

            new_states.append(final_state)
            expert_actions.append(expert_action)

            state_shape = np.shape(final_state)[0]
            model_output = model.predict(final_state.reshape(-1, state_shape))[0]
            action = self.list_to_GridAction(list(model_output))
            next_state, reward, done, info = self.env.step(state=state, action=action)
            cumulative_reward += reward
            state = deepcopy(next_state)
        logger.info(' Collected reward is: %d.' % (cumulative_reward))
        return new_states, expert_actions

    def add_new_data_in_experience(self, new_states, expert_actions):
        self.inputs = np.concatenate((self.inputs, np.array(new_states)), axis=0)
        if self.scale_outputs:
            self.outputs = np.concatenate((self.outputs, self.scaler.transform(np.array(expert_actions))), axis=0)
        else:
            self.outputs = np.concatenate((self.outputs, np.array(expert_actions)), axis=0)

    def list_to_GridAction(self, l):
        charge = []
        discharge = []
        generation = {g.name: 0. for g in self.grid.generators if g.steerable}
        for b in self.grid.storages:
            charge.append(l[0])
            l.pop(0)

        for b in self.grid.storages:
            discharge.append(l[0])
            l.pop(0)

        for g in self.grid.generators:
            if g.steerable:
                generation[g.name] = l[0] if l[0] >= 0.5 * g.min_stable_generation * g.capacity else 0.
                l.pop(0)

        assert (not l)
        return GridAction(generation, charge, discharge)

    def load_data(self):
        with open(self.path_to_stored_experience + "/" + self.env.simulator.case + "_optimization_experience_" + str(
                self.control_horizon) + ".p", "rb") as fp:
            self.states, self.forecasts, self.actions = pickle.load(fp)

    def process_data(self):
        list_X = []
        for state in self.states:
            values = decode_GridState(state, self.features)
            list_X.append(values)

        if self.use_forecasts:
            forecasts_list = []
            for forecast in self.forecasts:
                forecasts_list.append(np.concatenate(forecast))

            self.inputs = np.concatenate((np.array(list_X), np.array(forecasts_list)),
                                         axis=1)  # [consumption, soc1, soc2, ..., PV production, date]
        else:
            self.inputs = np.array(list_X)

        if self.scale_outputs:
            self.scaler = MinMaxScaler()
            max_generators = [g.capacity for g in self.grid.generators if g.steerable]
            max_storages_charge = [b.capacity for b in self.grid.storages]
            max_storages_discharge = [b.capacity for b in self.grid.storages]

            self.scaler.fit([np.zeros(len(self.actions[0])),
                             np.array(max_storages_charge + max_storages_discharge + max_generators)])

            self.outputs = self.scaler.transform(np.array(self.actions))
            # [charge1, charge2 ,..., discharge1, discharge2, ..., genset1, genset2, ...]
        else:
            self.outputs = np.array(self.actions)

    def create_models_from_dict(self, model_dict):
        self.sl_models = dict()
        for name, model in model_dict.items():
            self.sl_models[name] = eval(model)

    def store_model(self, model_name, model):
        with open(self.path_to_store_models + "/" + model_name + "_" + str(self.control_horizon) + ".p", "wb") as f:
            pickle.dump(model, f)


agent_type = SLAgent
