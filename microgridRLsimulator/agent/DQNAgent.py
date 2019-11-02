# -*- coding: utf-8 -*-
import multiprocessing
from abc import ABC

from microgridRLsimulator.agent.Stable_baseline_agent import SBAgent
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


class DQNAgent(SBAgent, ABC):

    def __init__(self, env, n_episodes, gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000,
                 learning_starts=1000, target_network_update_freq=500,
                 prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                 prioritized_replay_eps=1e-06, param_noise=False, n_test_episodes=1, options_filename=None,
                 load_path=None):
        super().__init__(env=env, n_episodes=n_episodes, options_filename=options_filename,
                         n_test_episodes=n_test_episodes, load_path=load_path)
        self.model = DQN(MlpPolicy, self.env, verbose=1, tensorboard_log="./microgrid_tensorboard/",
                         gamma=gamma, learning_rate=learning_rate, buffer_size=buffer_size,
                         exploration_fraction=exploration_fraction,
                         exploration_final_eps=exploration_final_eps, train_freq=train_freq, batch_size=batch_size,
                         checkpoint_freq=checkpoint_freq,
                         learning_starts=learning_starts, target_network_update_freq=target_network_update_freq,
                         prioritized_replay=prioritized_replay, prioritized_replay_alpha=prioritized_replay_alpha,
                         prioritized_replay_beta0=prioritized_replay_beta0,
                         prioritized_replay_eps=prioritized_replay_eps, param_noise=param_noise)

    @staticmethod
    def name():
        return "DQN"

    def load(self, model_name):
        self.model = DQN.load(model_name)


agent_type = DQNAgent
