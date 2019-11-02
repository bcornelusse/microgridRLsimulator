# -*- coding: utf-8 -*-
import multiprocessing
import os
from abc import ABC

import tensorflow as tf

from microgridRLsimulator.agent.Stable_baseline_agent import SBAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv


class PPOAgent(SBAgent, ABC):

    def __init__(self, env, n_episodes, reccurent_policies, gamma,
                 n_steps, ent_coef, learning_rate, vf_coef,
                 max_grad_norm, lam, nminibatches, noptepochs, cliprange, net_arch, options_filename=None, n_cpu=None,
                 n_test_episodes=1, load_path=None):
        super().__init__(env=env, n_episodes=n_episodes, options_filename=options_filename,
                         n_test_episodes=n_test_episodes, load_path=load_path)

        self.reccurent_policies = reccurent_policies
        self.n_cpu = multiprocessing.cpu_count() if n_cpu is None else n_cpu
        env = SubprocVecEnv([lambda: self.env for i in range(self.n_cpu)])
        if self.reccurent_policies:
            policy = "MlpLstmPolicy"
            # the number of environments run in parallel should be a multiple of nminibatches.
        else:
            policy = MlpPolicy
        self.model = PPO2(policy, env, verbose=1, tensorboard_log="./microgrid_tensorboard/", gamma=gamma,
                          n_steps=n_steps, ent_coef=ent_coef, learning_rate=learning_rate, vf_coef=vf_coef,
                          max_grad_norm=max_grad_norm, lam=lam, nminibatches=nminibatches, noptepochs=noptepochs,
                          cliprange=cliprange, cliprange_vf=None,
                          policy_kwargs={"net_arch": net_arch, "act_fun": tf.keras.activations.linear},
                          full_tensorboard_log=False)

    @staticmethod
    def name():
        return "PPO"

    def load(self, model_name):
        self.model = PPO2.load(model_name)


agent_type = PPOAgent
