# -*- coding: utf-8 -*-

import pandas as pd
import json
from datetime import datetime
import os
import itertools
from datetime import timedelta

from microgridRLsimulator.history import Database
from microgridRLsimulator.simulate.gridstate import GridState
from microgridRLsimulator.model.grid import Grid
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.plot import Plotter
from microgridRLsimulator.utils import positive, negative


class Simulator:
    def __init__(self, start_date, end_date, case, decision_horizon=1, results_folder=None, results_file=None):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        :param decision_horizon:
        :param results_folder: if None, set to default location
        :param results_file: if None, set to default file
        """

        this_dir, _ = os.path.split(__file__)
        package_dir = os.path.dirname(this_dir)
        parent_package_dir = os.path.dirname(package_dir)


        MICROGRID_CONFIG_FILE = os.path.join(package_dir, "data/%s.json" % case)
        MICROGRID_DATA_FILE = os.path.join(package_dir, 'data/%s_dataset.csv' % case)

        # setting results folder
        if results_folder is None:
            self.RESULTS_FOLDER = "results/results_%s_%s" % (
                case, datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            self.RESULTS_FOLDER = os.path.join(parent_package_dir, self.RESULTS_FOLDER)
        else:
            self.RESULTS_FOLDER = results_folder

        # setting results file
        if results_file is None:
            self.RESULTS_FILE = "%s/%s_out.json" % (self.RESULTS_FOLDER, case)
        else:
            self.RESULTS_FILE = results_file

        with open(MICROGRID_CONFIG_FILE, 'rb') as jsonFile:
            data = json.load(jsonFile)
            self.grid = Grid(data)

        self.case = case
        self.database = Database(MICROGRID_DATA_FILE, self.grid)
        self.actions = {}
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = pd.date_range(start=start_date, end=end_date, freq=str(decision_horizon) + 'h')
        self.high_level_actions = self._infer_high_level_actions()

        self.env_step = 0
        self.cumulative_cost = 0.
        self.grid_states = []

    def reset(self):
        """
        Resets the state of the simulator and returns a state representation for the agent.

        :return: A state representation for the agent as a list
        """
        self.actions = {}
        self.env_step = 0
        self.cumulative_cost = 0.

        # Initialize a gridstate
        self.grid_states = [GridState(self.grid, self.start_date)]

        realized_non_flexible_production = 0.0
        for g in self.grid.generators:
            if not g.steerable:
                realized_non_flexible_production += self.database.get_columns(g.name, self.start_date)

        realized_non_flexible_consumption = 0.0
        for l in self.grid.loads:
            realized_non_flexible_consumption += self.database.get_columns(l.name, self.start_date)

        # Add in the state the information about renewable generation and demand
        self.grid_states[-1].non_steerable_production = realized_non_flexible_production
        self.grid_states[-1].non_steerable_consumption = realized_non_flexible_consumption

        return self._decode_state(self.grid_states[-1])

    def step(self, state, high_level_action):
        """
        Method that can be called by an agent to create a transition of the system.

        :param state: state from which the transition occurs
        :param high_level_action: Action taken by an agent (translated later on into an implementable action)
        :return: a tuple (next_state, reward, termination_condition)
        """

        dt = self.date_range[self.env_step]

        # Use the high level action provided and the current state to generate the low level actions for each component
        actions = self._construct_action(high_level_action, state)

        # Record these actions in a json file
        self.actions[dt.strftime('%y/%m/%d_%H')] = actions.to_json()

        #  Update the step number and check the termination condition
        self.env_step += 1
        is_terminal = False
        if self.env_step == len(self.date_range) - 1:
            is_terminal = True
        p_dt = dt + timedelta(hours=1)

        # Construct an empty next state
        next_grid_state = GridState(self.grid, p_dt)

        ## Apply the control actions
        n_storages = len(self.grid.storages)

        # Compute next state of storage devices based on the control actions
        # and the storage dynamics
        next_soc = [0.0] * n_storages
        actual_charge = [0.0] * n_storages
        actual_discharge = [0.0] * n_storages
        for b in range(n_storages):
            (next_soc[b], actual_charge[b], actual_discharge[b]) = self.grid.storages[b].simulate(
                self.grid_states[-1].state_of_charge[b], actions.charge[b], actions.discharge[b]
            )
            # Store the computed level of storage to the next state
            next_grid_state.state_of_charge[b] = next_soc[b]

        # Record the control actions for the storage devices to the current state
        self.grid_states[-1].charge = actual_charge[:]
        self.grid_states[-1].discharge = actual_discharge[:]

        # Apply the control actions for the steerable generators based on the generators dynamics
        actual_generation = {g: 0. for g in self.grid.generators}
        actual_generation_cost = {g: 0. for g in self.grid.generators}

        for g in self.grid.generators:
            if g.steerable:
                actual_generation[g], actual_generation_cost[g] = g.simulate_generator(
                    actions.conventional_generation[g])
        # Record the generation output to the current state
        self.grid_states[-1].generation = list(actual_generation.values())

        # Deduce actual production and consumption based on the control actions taken and the 
        # actual components dynamics
        actual_production = self.grid_states[-1].non_steerable_production \
                            + sum(actual_discharge[b] for b in range(n_storages)) \
                            + sum(actual_generation[g] for g in self.grid.generators)
        actual_consumption = self.grid_states[-1].non_steerable_consumption \
                             + sum(actual_charge[b] for b in range(n_storages))

        # Store the total production and consumption
        self.grid_states[-1].production = actual_production
        self.grid_states[-1].consumption = actual_consumption

        # Perform the final balancing
        actual_import = actual_export = 0
        net_import = actual_consumption - actual_production

        if positive(net_import):
            actual_import = net_import * self.grid.period_duration
        elif negative(net_import):
            actual_export = -net_import * self.grid.period_duration

        # NOTE: for now since the system is off grid we assume that:
        # a) Imports are equivalent to load shedding
        # b) exports are equivalent to production curtailment
        self.grid_states[-1].grid_import = actual_import
        self.grid_states[-1].grid_export = actual_export

        # Compute the final cost of operation as the sum of three terms:
        # a) fuel costs for the generation
        # b) curtailment cost for the excess of generation that had to be curtailed
        # c) load shedding cost for the excess of load that had to be shed in order to maintain balance in the grid
        # Note that we can unbundle the individual costs according to the objective optimized
        self.grid_states[-1].fuel_cost = sum(actual_generation_cost[g] for g in self.grid.generators)
        self.grid_states[-1].curtailment_cost = actual_export * self.grid.curtailement_price
        self.grid_states[-1].load_not_served_cost = actual_import * self.grid.load_shedding_price

        self.grid_states[-1].total_cost = self.grid_states[-1].load_not_served_cost + self.grid_states[
            -1].curtailment_cost + self.grid_states[-1].fuel_cost

        self.cumulative_cost += self.grid_states[-1].total_cost
        self.grid_states[-1].cum_total_cost = self.cumulative_cost

        # Add in the next state the information about renewable generation and demand
        # Note: here production refers only to the non-steerable production
        realized_non_flexible_production = 0.0
        for g in self.grid.generators:
            if not g.steerable:
                realized_non_flexible_production += self.database.get_columns(g.name, p_dt)

        realized_non_flexible_consumption = 0.0
        for l in self.grid.loads:
            realized_non_flexible_consumption += self.database.get_columns(l.name, p_dt)

        next_grid_state.non_steerable_production = realized_non_flexible_production
        next_grid_state.non_steerable_consumption = realized_non_flexible_consumption
        self.grid_states.append(next_grid_state)

        # Pass the information about the next state, cost of the previous control actions and termination condition 
        return self._decode_state(next_grid_state), -self.grid_states[-2].total_cost, is_terminal

    def store_and_plot(self, learning_results=None):
        """
        Store and plot results.

        :param learning_results: A list containing the results of the learning progress
        :return: Nothing.
        """

        results = dict(dates=["%s" % d.date_time for d in self.grid_states],
                       soc=[d.state_of_charge for d in self.grid_states],
                       charge=[d.charge for d in self.grid_states],
                       discharge=[d.discharge for d in self.grid_states],
                       generation=[d.generation for d in self.grid_states],
                       cum_total_cost=[d.cum_total_cost for d in self.grid_states],
                       energy_cost=[d.total_cost for d in self.grid_states],
                       production=[d.production for d in self.grid_states],
                       consumption=[d.consumption for d in self.grid_states],
                       non_steerable_production=[d.non_steerable_production for d in self.grid_states],
                       non_steerable_consumption=[d.non_steerable_consumption for d in self.grid_states],
                       grid_import=[d.grid_import for d in self.grid_states],
                       grid_export=[d.grid_export for d in self.grid_states],
                       avg_rewards=learning_results)

        if not os.path.isdir(self.RESULTS_FOLDER):
            os.makedirs(self.RESULTS_FOLDER)

        with open(self.RESULTS_FILE, 'w') as jsonFile:
            json.dump(results, jsonFile)

        plotter = Plotter(results, '%s/%s' % (self.RESULTS_FOLDER, self.case))
        plotter.plot_results()

    def _infer_high_level_actions(self):
        """
        Method that infers the full high-level action space by the number of controllable storage devices. Simultaneous
        charging of a battery and charging of another is ruled-out from the action set

        :return: list of possible tuples
        """

        # The available decisions for the storage device are charge (C), discharge (D) and idle (I)
        combinations_list = itertools.product('CDI', repeat=len(self.grid.storages))

        # Generate the total actions set but exclude simultaneous charging of one battery and discharging of another
        combos_exclude_simul = list(filter(lambda x: not ('D' in x and 'C' in x), combinations_list))

        return combos_exclude_simul

    def _decode_state(self, gridstate):
        """
        Method that transforms the grid state into a list that contains the important information for the decision
        making process.

        :param gridstate: Gridstate object that contains the whole information about the micro-grid
        :return: list of the type [ non_flex_consumption , [state_of_charge_0, state_of_charge_1,...] , non_flex_production]
        """

        return [gridstate.non_steerable_consumption, gridstate.state_of_charge, gridstate.non_steerable_production] # TODO AgentState object?

    def _construct_action(self, high_level_action, state):
        """
        Maps the  high level action provided by the agent into an action implementable by the simulator.
        high_level_action : 0 --> charging bat 1 - charging bat 2 ...
        high_level_action : 1 --> charging bat 1 - idling bat 2 ...
        high_level_action : 2 --> idle
        ...
        """

        n_storages = len(self.grid.storages)

        generation = {g: 0. for g in self.grid.generators}
        charge = [0. for b in range(n_storages)]
        discharge = [0. for b in range(n_storages)]

        consumption = state[0]
        state_of_charge = state[1]
        non_flex_production = state[2]

        d = self.grid.period_duration
        # Compute the residual generation : 
        # a) if it is positive there is excess of energy
        # b) if it is negative there is deficit
        net_generation = non_flex_production - consumption

        if positive(net_generation):
            storages_to_charge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "C"]
            # if there is excess, charge the storage devices that are in Charge mode by the controller
            for b in storages_to_charge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                empty_space = storage.capacity - soc
                charge[b] = min(empty_space / d, net_generation / d, storage.max_charge_rate)
                net_generation -= charge[b]
        elif negative(net_generation):
            storages_to_discharge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "D"]
            # if there is deficit, discharge the storage devices that are in Discharge mode by the controller
            for b in storages_to_discharge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                discharge[b] = min(soc / d, -net_generation / d, storage.max_charge_rate)
                net_generation += discharge[b]
            # Use the steerable generation to supply the remaining deficit
            for g in self.grid.generators:
                if g.steerable:
                    generation[g] = -net_generation

        return GridAction(generation, charge, discharge)