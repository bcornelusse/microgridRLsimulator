
from microgridRLsimulator.agent.agent import Agent
from copy import deepcopy
import numpy as np
import os
from datetime import datetime

# import numpy as np
# from copy import deepcopy
from microgridRLsimulator.utils import time_string_for_storing_results


class SBAgent(Agent):

    def __init__(self, env, n_episodes, options_filename=None,
                 n_test_episodes=1, load_path=None):
        super().__init__(env)
        self.n_episodes = n_episodes
        self.n_test_episodes = n_test_episodes
        self.path_to_store_models = "models/" + self.name() + "/" + env.simulator.case + "/" + options_filename + "/" + "trained_from_" + env.simulator.start_date.strftime(
            "%m-%d-%Y") + "_to_" + env.simulator.end_date.strftime("%m-%d-%Y")
        self.model_name = self.path_to_store_models + '/best_model_' + datetime.now().strftime('%Y-%m-%d_%H%M') + '.pkl'
        if not os.path.isdir(self.path_to_store_models):
            os.makedirs(self.path_to_store_models)
        self.options_filename = options_filename
        self.best_value = -np.inf
        self.model = None
        self.load_path = load_path

    @staticmethod
    def name():
        return "SBA"

    def train_agent(self):
        timesteps_per_episode = self.env.simulator.date_range.shape[0] - 1
        total_timesteps = self.n_episodes * timesteps_per_episode
        self.model.learn(total_timesteps=total_timesteps, tb_log_name=self.name(), callback=self.callback)


    def simulate_agent(self, agent_options=None):
        if self.load_path is not None:
            self.load(self.load_path) # Load the model specified in the path
        else:
            try:
                self.load(self.model_name) # Load best model
            except:
                pass # Use last model

        for i in range(1, self.n_test_episodes + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False
            while not done:

                action, _states = self.model.predict(state)
                next_state, reward, done, info = self.env.step(state=state, action=action)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation: %d and the reward is: %d.' % (i, cumulative_reward))
            self.env.simulator.store_and_plot(
                folder="results/" + self.name() + "/" + self.env.simulator.case + "/" + self.options_filename + "/" + time_string_for_storing_results(
                    self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime(
                        "%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                    self.env.simulator.case) + "_" + str(i), agent_options=agent_options)

    def callback(self, _locals, _globals):
        if np.mean(_locals['self'].episode_reward) >= self.best_value:
            _locals['self'].save(self.model_name)
            self.best_value = np.mean(_locals['self'].episode_reward)
        return True


