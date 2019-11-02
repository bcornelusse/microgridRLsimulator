from microgridRLsimulator.agent.agent import Agent
from collections import namedtuple
from collections import defaultdict
import math
import random
from microgridRLsimulator.simulate.simulatorMCTS import SimulatorMCTS
from microgridRLsimulator.utils import time_string_for_storing_results

microgrid_node = namedtuple("microgrid_node", "State action terminal")


class node(microgrid_node):
    pass


class MCTSAgent(Agent):

    def __init__(self, env, n_test_episodes=1, exploration_weight=40.414, deviation_factor=0.50):
        super().__init__(env)
        self.deviation_factor = deviation_factor
        self.n_test_episodes = n_test_episodes
        self.children = dict()
        self.best_move = dict()
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.childrens = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.current_state_env_step = defaultdict(int)
        self.current_state_grid = defaultdict(int)
        self.cumulative_reward = 0
        self.env.simulator.reset()
        self.a = SimulatorMCTS(self.env.simulator.start_date, self.env.simulator.end_date, self.env.simulator.env_step,
                          self.env.simulator.grid_states[-1], self.env.simulator.case, self.deviation_factor)

    @staticmethod
    def name():
        return "MCTS"

    def train_agent(self):
        pass

    def simulate_agent(self, agent_options=None):
        for i in range(1, self.n_test_episodes + 1):
            self.env.reset()
            state = node(tuple([0, -1]), 0, False)
            for _ in range(len(self.env.simulator.date_range) - 1):

                for rollout_time in range(20):
                    self.do_rollout(state)

                optimal_move = self.choose(state)
                state = optimal_move
                optimal_action = optimal_move.action
                next_state, reward, done, info = self.env.step(state=1, action=optimal_action)
                self.a = SimulatorMCTS(self.env.simulator.start_date, self.env.simulator.end_date,
                                       self.env.simulator.env_step,
                                       self.env.simulator.grid_states[-1], self.env.simulator.case,
                                       self.deviation_factor)
                print('optimal reward got:', reward)
                self.cumulative_reward += reward
            self.env.simulator.store_and_plot(
                folder="results/" + self.name() + "/" + self.env.simulator.case + "/" + time_string_for_storing_results(
                    self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime(
                        "%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                    self.env.simulator.case) + "_" + str(i), agent_options=agent_options)




    def find_children(self, state):
        if state.terminal:
            return
        path = list(state.State)
        # print(path)
        children = set()
        actions = list(range(len(self.env.simulator.high_level_actions)))
        for i in actions:
            path.append(i)
            if path[-2] + 1 <= len(self.env.simulator.date_range) - 2:
                path.append(path[-2] + 1)
                children.add(node(tuple(path), i, False))

            else:
                path.append(path[-2] + 1)
                children.add(node(tuple(path), i, True))
            path = list(state.State)

        return children

    def select(self, state):
        path = []

        while True:
            path.append(state)

            if state not in self.childrens or not self.childrens[state]:
                # print('same state returned')
                return path

            unexplored = set(self.childrens[state]) - set(self.childrens.keys())
            # print('this is the unexplored:', unexplored)
            if unexplored:

                if list(unexplored)[0].terminal:
                    # print('inside terminal')
                    unexplored = list(unexplored)
                    path.append(random.choice(unexplored))
                    return path
                else:
                    n = unexplored.pop()
                    path.append(n)
                    return path

            state = self.uct_select(state)
            # print('state slelected by uct', state)

    def uct_select(self, state):

        assert all(n in self.childrens for n in self.childrens[state])

        log_N_vertex = 2 * math.log(self.N[state])

        def uct(n):
            "Upper confidence bound for trees"

            # print('uct--n', n, self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n]),
            # 'Q[n]', self.Q[n], 'N[n]', self.N[n])
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

        state = max(self.childrens[state], key=uct)

        return state

    def expand(self, leaf, path):


        if leaf in self.childrens:
            return
        # print('node to expand:', leaf)
        self.childrens[leaf] = self.find_children(leaf)

    def random_moves(self, path):
        self.a.reset()

        all_actions = list(range(len(self.env.simulator.high_level_actions)))
        total_reward = 0
        for state in path[1:]:
            # print('path to add reward of toll now', state)
            # print(state.action)
            if state.terminal:
                return total_reward
            nex_state, reward_simulator, done = self.a.step(state.action)
            # print('reward',reward_simulator)
            total_reward += (reward_simulator / 100)

        state_random = path[-1]
        if state_random.terminal:
            return total_reward

        for i in range(3):

            try:
                action = random.choice(all_actions)
                nex_state, reward, done = self.a.step(action)
                # print('random action chosen:', action)
                # print('reward by random action',reward)

                total_reward += (reward / 100)

            except:
                # print('inside exception of reward')
                return total_reward
        return total_reward

    def backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            # print(node)
            self.N[node] += 1

            self.Q[node] += reward
        # print(self.N)
        # print(self.Q)
        # print('backpropagation completed')

    def choose(self, state):

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.childrens[state], key=score)

    def do_rollout(self, state):

        path = self.select(state)
        leaf = path[-1]
        # print(leaf.terminal)
        if leaf.terminal == False:
            self.expand(leaf, path)
        reward = self.random_moves(path)
        self.backpropagate(path, reward)

agent_type = MCTSAgent



