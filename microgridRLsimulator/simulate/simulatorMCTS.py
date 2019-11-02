import pandas as pd
import numpy as np
from itertools import chain
import json
from datetime import datetime
import os
import itertools
from datetime import timedelta
from dateutil.parser import isoparse

from microgridRLsimulator.history import Database
from microgridRLsimulator.simulate.gridstate import GridState
from microgridRLsimulator.model.grid import Grid
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.plot import Plotter
from microgridRLsimulator.utils import positive, negative, decode_GridState
from microgridRLsimulator.simulate.forecaster import Forecaster
from microgridRLsimulator.simulate import Simulator


class SimulatorMCTS():
    def __init__(self, start_date, end_date, env_step, grid_states, case, deviation_factor=None):

        self.deviation_factor = deviation_factor
        self.reset_env_step = env_step
        self.reset_grid_state = grid_states
        self.env_step = env_step
        self.grid_states = grid_states
        this_dir, _ = os.path.split(__file__)
        package_dir = os.path.dirname(this_dir)
        MICROGRID_CONFIG_FILE = os.path.join(package_dir, "data", case, "%s.json" % case)
        with open(MICROGRID_CONFIG_FILE, 'rb') as jsonFile:
            self.data = json.load(jsonFile)
            self.grid = Grid(self.data)
            self.objectives = self.data["objectives"]
        self.state_features = self.data["features"]
        self.backcast_steps = self.data["backcast_steps"]

        self.MICROGRID_DATA_FILE = os.path.join(package_dir, "data", case, '%s_dataset.csv' % case)

        self.date_range = pd.date_range(start=start_date, end=end_date,
                                        freq=str(int(self.grid.period_duration * 60)) + 'min')
        self.control_horizon = (len(self.date_range) - self.env_step + 1)
        self.df = pd.read_csv(self.MICROGRID_DATA_FILE, sep=";|,", parse_dates=True, index_col='DateTime',
                              engine='python')

        self.database = Database(self.MICROGRID_DATA_FILE, self.grid)
        self.forecast_date_range = list(sorted(set(chain(self.date_range, pd.date_range(start=self.date_range[-1],
                                                                                        periods=self.control_horizon,
                                                                                        freq=self.date_range.freq)))))
        self.high_level_actions = self._infer_high_level_actions()
        self.cumulative_cost = 0

        std_factor = np.linspace(.0, self.deviation_factor,
                                 num=self.control_horizon)  # increasing from 0 at current step to deviation at final step
        for i in range(self.control_horizon):

            for g in self.grid.generators:
                if not g.steerable:
                    time = (self.env_step + i) * self.grid.period_duration * 60
                    updated_capacity = g.find_capacity(time)
                    # print(g.name, self.df[g.name][self.forecast_date_range[self.env_step + i]])
                    non_flexible_production = self.database.get_columns(g.name,
                                                                        self.forecast_date_range[self.env_step + i]) * (
                                                          updated_capacity / g.initial_capacity)
                    self.df[g.name][
                        self.forecast_date_range[self.env_step + i]] = non_flexible_production + np.random.normal(
                        scale=std_factor[i] * non_flexible_production)
                    # print(g.name,self.forecast_date_range[self.env_step + i],'changed to',self.df[g.name][self.forecast_date_range[self.env_step + i]])

            for l in self.grid.loads:
                non_flexible_consumption = self.database.get_columns(l.name,
                                                                     self.forecast_date_range[self.env_step + i])
                # print(l.name,self.df[l.name][self.forecast_date_range[self.env_step + i]])
                self.df[l.name][
                    self.forecast_date_range[self.env_step + i]] = non_flexible_consumption + np.random.normal(
                    scale=std_factor[i] * non_flexible_consumption)
                # print(l.name, self.forecast_date_range[self.env_step + i], 'changed to',self.df[l.name][self.forecast_date_range[self.env_step + i]])

            # self.database = Database(self.MICROGRID_DATA_FILE, self.grid)

    def reset(self):
        self.env_step = self.reset_env_step
        self.grid_states = self.reset_grid_state

    def step(self, high_level_action=None, low_level_action=None):
        """
        Method that can be called by an agent to create a transition of the system.

        :param high_level_action: Action taken by an agent (translated later on into an implementable action)
        :return: a tuple (next_state, reward, termination_condition)
        """
        dt = self.date_range[self.env_step]
        # Use the high level action provided and the current state to generate the low level actions for each component
        if high_level_action is not None:
            actions = self._construct_action(high_level_action)
        # Or provide directly low level actions
        elif low_level_action is not None:
            actions = low_level_action if isinstance(low_level_action, GridAction) \
                else self._construct_action_from_list(low_level_action)

        #  Update the step number and check the termination condition
        self.env_step += 1
        is_terminal = False
        if self.env_step == len(self.date_range) - 1:
            is_terminal = True
        p_dt = self.date_range[self.env_step]  # It's the next step

        # Construct an empty next state
        next_grid_state = GridState(self.grid, p_dt)

        # Apply the control actions
        n_storages = len(self.grid.storages)

        # Compute next state of storage devices based on the control actions
        # and the storage dynamics
        next_soc = [0.0] * n_storages
        actual_charge = [0.0] * n_storages
        actual_discharge = [0.0] * n_storages
        for b in range(n_storages):
            (next_soc[b], actual_charge[b], actual_discharge[b]) = self.grid.storages[b].simulate(
                self.grid_states.state_of_charge[b], actions.charge[b], actions.discharge[b],
                self.grid.period_duration
            )
            # Store the computed capacity and level of storage to the next state
            next_grid_state.capacities[b] = self.grid.storages[b].capacity
            next_grid_state.n_cycles[b] = self.grid.storages[b].n_cycles
            next_grid_state.state_of_charge[b] = next_soc[b]

        # Record the control actions for the storage devices to the current state
        self.grid_states.charge = actual_charge[:]
        self.grid_states.discharge = actual_discharge[:]

        # Apply the control actions for the steerable generators based on the generators dynamics
        actual_generation = {g: 0. for g in self.grid.generators}
        actual_generation_cost = {g: 0. for g in self.grid.generators}

        for g in self.grid.generators:
            if g.steerable:
                actual_generation[g], actual_generation_cost[g] = g.simulate_generator(
                    actions.conventional_generation[g.name], self.grid.period_duration)
        # Record the generation output to the current state
        self.grid_states.generation = list(actual_generation.values())

        # Deduce actual production and consumption based on the control actions taken and the
        # actual components dynamics
        actual_production = self.grid_states.non_steerable_production \
                            + sum(actual_discharge[b] for b in range(n_storages)) \
                            + sum(actual_generation[g] for g in self.grid.generators)
        actual_consumption = self.grid_states.non_steerable_consumption \
                             + sum(actual_charge[b] for b in range(n_storages))

        # Store the total production and consumption
        self.grid_states.production = actual_production
        self.grid_states.consumption = actual_consumption

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
        self.grid_states.grid_import = actual_import
        self.grid_states.grid_export = actual_export

        # Compute the final cost of operation as the sum of three terms:
        # a) fuel costs for the generation
        # b) curtailment cost for the excess of generation that had to be curtailed
        # c) load shedding cost for the excess of load that had to be shed in order to maintain balance in the grid
        # Note that we can unbundle the individual costs according to the objective optimized
        self.grid_states.fuel_cost = sum(actual_generation_cost[g] for g in self.grid.generators)
        self.grid_states.curtailment_cost = actual_export * self.grid.curtailment_price
        self.grid_states.load_not_served_cost = actual_import * self.grid.load_shedding_price
        self.grid_states.total_cost = self.grid_states.load_not_served_cost + \
                                      self.grid_states.curtailment_cost + self.grid_states.fuel_cost

        multiobj = {'total_cost': self.grid_states.total_cost,
                    'load_shedding': actual_import,
                    'fuel_cost': self.grid_states.fuel_cost,
                    'curtailment': actual_export,
                    'storage_maintenance': {self.grid.storages[b].name: self.grid.storages[b].n_cycles for b in
                                            range(n_storages)}
                    }

        self.cumulative_cost += self.grid_states.total_cost
        next_grid_state.cum_total_cost = self.cumulative_cost
        # Add in the next state the information about renewable generation and demand
        # Note: here production refers only to the non-steerable production
        realized_non_flexible_production = 0.0
        for g in self.grid.generators:
            if not g.steerable:
                time = self.env_step * self.grid.period_duration * 60  # time in min (in order to be able to update capacity all min)
                g.update_capacity(time)
                realized_non_flexible_production += self.df[g.name][p_dt] * (
                        g.capacity / g.initial_capacity)
        next_grid_state.res_gen_capacities = [g.capacity for g in self.grid.generators if not g.steerable]

        realized_non_flexible_consumption = 0.0
        for l in self.grid.loads:
            realized_non_flexible_consumption += self.df[l.name][p_dt]

        next_grid_state.non_steerable_production = realized_non_flexible_production
        next_grid_state.non_steerable_consumption = realized_non_flexible_consumption
        self.grid_states = next_grid_state
        # Pass the information about the next state, cost of the previous control actions and termination condition
        return self._decode_state(self.grid_states), self._compute_rewards(
            multiobj), is_terminal  # +1 to take into account the current state

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

    def _decode_state(self, gridstates):
        """
        Method that transforms the grid state into a list that contains the important information for the decision
        making process.

        :param gridstates: a list of Gridstate object that contains the whole information about the micro-grid
        :return: list with default or selected  state features.
        """

        state_list = decode_GridState([gridstates], self.state_features,
                                      self.backcast_steps + 1)  # +1 because of the current state

        return state_list

    def _construct_action(self, high_level_action):
        """
        Maps the  high level action provided by the agent into an action implementable by the simulator.
        high_level_action : 0 --> charging bat 1 - charging bat 2 ...
        high_level_action : 1 --> charging bat 1 - idling bat 2 ...
        high_level_action : 2 --> idle
        ...
        """

        n_storages = len(self.grid_states.state_of_charge)

        generation = {g.name: 0. for g in self.grid.generators if g.steerable}
        charge = [0. for b in range(n_storages)]
        discharge = [0. for b in range(n_storages)]
        consumption = self.grid_states.non_steerable_consumption
        state_of_charge = self.grid_states.state_of_charge
        non_flex_production = self.grid_states.non_steerable_production

        d = self.grid.period_duration
        # Compute the residual generation :
        # a) if it is positive there is excess of energy
        # b) if it is negative there is deficit
        net_generation = non_flex_production - consumption
        if negative(net_generation):
            # check if genset has to be active, if it is the case: activation at the min stable capacity and update the net generation.
            total_possible_discharge = 0.0
            genset_total_capacity = 0.0
            genset_min_stable_total_capacity = 0.0
            storages_to_discharge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "D"]
            for b in storages_to_discharge:
                storage = self.grid.storages[b]
                total_possible_discharge += min(state_of_charge[b] * storage.discharge_efficiency / d,
                                                storage.max_discharge_rate)

            for g in self.grid.generators:  # TODO sort generators
                if g.steerable:
                    if net_generation + total_possible_discharge + genset_total_capacity < 0:
                        genset_min_stable_total_capacity += g.min_stable_generation * g.capacity
                        genset_total_capacity += g.capacity
                        generation[g.name] = g.min_stable_generation * g.capacity

            net_generation += genset_min_stable_total_capacity  # net generation takes into account the min stable production
        if positive(net_generation):
            storages_to_charge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "C"]
            # if there is excess, charge the storage devices that are in Charge mode by the controller
            for b in storages_to_charge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                empty_space = storage.capacity - soc
                charge[b] = min(empty_space / (d * storage.charge_efficiency), net_generation,
                                storage.max_charge_rate)  # net_generation is already in kW
                net_generation -= charge[b]
        elif negative(net_generation):
            storages_to_discharge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "D"]
            # if there is deficit, discharge the storage devices that are in Discharge mode by the controller
            for b in storages_to_discharge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                discharge[b] = min(soc * storage.discharge_efficiency / d, -net_generation,
                                   storage.max_discharge_rate)  # discharge = soc/d is always changed to soc*effi/d in the simulation
                net_generation += discharge[b]

            for g in self.grid.generators:  # TODO sort generators
                if g.steerable:
                    additional_generation = min(-net_generation, g.capacity - generation[g.name])
                    generation[g.name] += additional_generation  # Update the production of generator g
                    net_generation += additional_generation  # Update the remaining power to handle
        return GridAction(generation, charge, discharge)

    def _construct_action_from_list(self, actions_list):

        n_storages = len(self.grid.storages)
        generation = {g.name: 0. for g in self.grid.generators if g.steerable}
        charge = actions_list[:n_storages]
        discharge = actions_list[n_storages:2 * n_storages]
        gen = actions_list[2 * n_storages:]

        i = 0
        for g in self.grid.generators:
            if g.steerable:
                generation[g.name] = gen[i]
                i += 1

        return GridAction(generation, charge, discharge)

    def _compute_rewards(self, multiobj_dict):

        rewards_dict = {}
        for o, val in self.objectives.items():
            if val:
                rewards_dict[o] = multiobj_dict[o]
        if len(
                rewards_dict) == 1:  # I think it is better to always return a dict than sometimes a dict and sometimes a value
            return -list(rewards_dict.values())[0]
        else:
            return rewards_dict
