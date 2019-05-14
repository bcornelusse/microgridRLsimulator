# -*- coding: utf-8 -*-

from microgridRLsimulator.agent.agent import Agent

import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import logging

logger = logging.getLogger(__name__)  #: Logger.


class DQNAgent(Agent):
    """
    Agent implementing deep Q learning.
    """

    def __init__(self, simulation_environmnent, learning_rate=0.0001, eps=0.1, eps_decay=0.9986,
                 n_episodes=1000, gamma=0.99, batch_size=16, simulate_model=None, tensorboard_dir=None):
        """

        :param simulation_environmnent: Instance of Simulator
        :param learning_rate: gradient descent parameter
        :param eps: epsilon for epsilon-greedy policy
        :param eps_decay: Decay factor for eps
        :param n_episodes: how many episodes we train for
        :param gamma: discount factor
        :param batch_size: size of the set we want to train from
        :param simulate_model: path to a serialized model to be simulated (cancels the training phase).
        :param tensorboard_dir: folder path where to store logs if you want to use tensorboard.
        """

        super().__init__(simulation_environmnent)

        self.learning_rate = learning_rate
        self.eps = eps
        self.eps_decay = eps_decay
        self.episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.tensorboard_dir = tensorboard_dir
        self.model = self._build_model()
        self.long_term_memory = []
        self.average_reward = []
        self.simulate_model = simulate_model

    @staticmethod
    def name():
        return "DQN"

    def run(self):
        self.train_agent(refresh=True)
        self.simulate_agent()

    def train_agent(self, refresh=False, save_model=False):
        """

        :param refresh: use experience replay or not.
        :param save_model: Serialize the learned model to folder "models"
        :return: Nothing, trains self.model
        """

        if self.simulate_model is not None:
            # Do not train, just simulate
            return

        average_cumulative_reward = 0.0
        for i in range(self.episodes):
            # Initialize the environment
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False
            self.eps *= self.eps_decay
            short_term_memory = []

            while not done:
                # Refactor the state into a flat array (for convenience)
                state_array = self.state_refactoring(state)

                # Take e-greedy actions based
                if random.random() < self.eps:
                    action = random.randrange(0, self.num_actions)
                else:
                    # the flat array is used for the forward pass in the NN
                    action = np.argmax(self.model.predict(state_array.reshape(-1, self.state_size, 1)))

                # Get to the next state based on the high level action selected
                next_state, reward, done = self.env.step(state, action)
                #reward = self.reward_function(reward_info)

                # Refactor the new state into a flat array (for convenience)
                next_state_array = self.state_refactoring(next_state)

                cumulative_reward += reward
                logger.info(state_array)
                logger.info(self.model.predict(state_array.reshape(-1, self.state_size, 1)))

                # Update the short term memory for the online learning
                short_term_memory.append([state_array, action, reward, next_state_array, done])

                # Update the long term memory for the experience replay
                self.long_term_memory.append([state_array, action, reward, next_state_array, done])

                state = deepcopy(next_state)
                if done:
                    logger.info('i am in episode: %d and the reward is: %.2f with a value of eps: %.2f' % (
                        i, cumulative_reward, self.eps))

            # Create some plots every 1500 episodes to visualise the progress of training
            if i % 1500 == 0:
                self.env.store_and_plot()

            # On-line training of the NN
            if len(short_term_memory) > self.batch_size:
                self._train(short_term_memory)

            # Off-line training of the NN every 1000 episodes
            if i % 1000 == 0 and refresh:
                self._experience_replay(self.long_term_memory)

            # Smoothing
            average_cumulative_reward *= 0.9
            average_cumulative_reward += 0.1 * cumulative_reward
            self.average_reward.append(average_cumulative_reward)

        if save_model:
            self.model.save('models/MGmodel%s.tfl' % (int(time.time())))

    def simulate_agent(self, simulation_steps=1):
        """

        :return: Plots and stores results in files.
        """

        if self.simulate_model is not None:
            self.model.load(self.simulate_model, weights_only=True)

        for i in range(1, simulation_steps + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                state_array = self.state_refactoring(state)
                action = np.argmax(self.model.predict(state_array.reshape(-1, self.state_size, 1)))
                next_state, reward, done = self.env.step(state, action)
                #reward = self.reward_function(reward_info)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation: %d and the reward is: %d.' % (i, cumulative_reward))
        # Pass the progress of the cumulative reward in order to plot the learning progress
        self.env.store_and_plot(learning_results=self.average_reward)

    def _train(self, memory, n_epochs=10):
        """
        Train the NN.
        :param memory: Set of transitions as input for learning.
        :param n_epochs: Number of epochs.
        :return: Nothing, trains self.model
        """

        # Generate the targets-features for on-line training
        replay_ids = list(np.random.choice(range(len(memory)), self.batch_size))
        targets = np.zeros((self.batch_size, self.num_actions))
        inputs = np.zeros((self.batch_size, self.state_size))
        train_data = []
        for j in replay_ids:
            st, a, r, st_t, terminal = memory[j]
            targets[replay_ids.index(j)] = self.model.predict(st.reshape(-1, self.state_size, 1))
            Q_sa = self.model.predict(st_t.reshape(-1, self.state_size, 1))
            inputs[replay_ids.index(j):replay_ids.index(j) + 1] = st
            if terminal:
                targets[replay_ids.index(j), a] = r
            else:
                targets[replay_ids.index(j), a] = r + self.gamma * np.max(Q_sa)
            train_data.append([inputs[replay_ids.index(j)], targets[replay_ids.index(j)]])
        X = np.array([i[0] for i in train_data]).reshape(-1, self.state_size, 1)
        y = [i[1] for i in train_data]
        self.model.fit({'input': X}, {'targets': y}, n_epoch=n_epochs, show_metric=True, run_id='Main')

    def _experience_replay(self, long_memory, n_epochs=5):
        """

        :param long_memory: Set of transitions
        :param n_epochs: Number of epochs.
        :return:
        """

        # Generate the targets-features for off-line training
        replay = long_memory
        targets = np.zeros((len(long_memory), self.num_actions))
        inputs = np.zeros((len(long_memory), self.state_size))
        train_data = []
        for j in range(len(long_memory)):
            st, a, r, st_t, terminal = replay[j]
            targets[j] = self.model.predict(st.reshape(-1, self.state_size, 1))
            Q_sa = self.model.predict(st_t.reshape(-1, self.state_size, 1))
            inputs[j:j + 1] = st
            if terminal:
                targets[j, a] = r
            else:
                targets[j, a] = r + self.gamma * np.max(Q_sa)
            train_data.append([inputs[j], targets[j]])
        X = np.array([i[0] for i in train_data]).reshape(-1, self.state_size, 1)
        y = [i[1] for i in train_data]
        self.model.fit({'input': X}, {'targets': y}, n_epoch=n_epochs, show_metric=True, run_id='Main')

    def _build_model(self, hidden_layer_nodes=12):
        """
        Q function model

        :param hidden_layer_nodes:
        :return: the created model as a NN.
        """

        network = input_data(shape=[None, self.state_size, 1], name='input')
        network = fully_connected(network, hidden_layer_nodes, activation='relu')
        network = fully_connected(network, self.num_actions, activation='linear', weights_init='zeros')
        network = regression(network, optimizer='adam', learning_rate=self.learning_rate, loss='mean_square',
                             name='targets')
        if self.tensorboard_dir is not None:
            model = tflearn.DNN(network, tensorboard_dir=self.tensorboard_dir)
        else:
            model = tflearn.DNN(network)
        return model

    def reward_function(self, reward_info):
        """
        Method that transforms the reward infos into a reward value with the help of a reward function tuned by the user.

        :param reward_info: dictionary that contains reward information relative to the chosen objectives 
        (total_cost, fuel_cost, load_shedding, curtailment, storage_maintenance).
        :return: reward value from a tuned reward function.
        """
        reward = - reward_info["total_cost"]
        return reward

    def plot_progress(self):
        plt.figure()
        plt.title('episodes: %d lr: %.4f' % (self.episodes, self.learning_rate))
        plt.plot(range(len(self.average_reward)), self.average_reward)
        plt.show()

agent_type = DQNAgent
