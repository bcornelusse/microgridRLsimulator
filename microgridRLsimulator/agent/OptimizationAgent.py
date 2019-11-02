import os

import json

from microgridRLsimulator.agent.agent import Agent
import logging
import time
from copy import deepcopy
from pyomo.environ import *
from pyomo.opt import SolverFactory

from microgridRLsimulator.simulate.forecaster import Forecaster
from microgridRLsimulator.simulate.gridaction import GridAction
import pickle

#  TODO Add constraint: when there are several storages, prevent the simulatenous charge and discharge of different
#  batteries The only differences between optimization lookahead and simulation are the presence of tolerances on net
#  import and capacity update
from microgridRLsimulator.utils import time_string_for_storing_results

logger = logging.getLogger(__name__)  #: Logger


class OptimizationAgent(Agent):
    """
    Implements an optimizaton controller
    """
    def __init__(self, env, control_horizon, simulation_horizon, options_filename=None, save_data=False,
                 path_to_store_experience=None,
                 forecast_type="exact", n_test_episodes=1):
        """
        :param: env: a microgrid environment
        :param: control_horizon: the number of lookahead steps in the optimization model.
        :param: simulation_horizon: the number of optimized actions that will be simulated.
        :param: options_filename: the agent options filename.
        :param: save_data: flag to save or not the data.
        :param: path_to_score_experience: the path where experiences are saved.
        :param: forecast_type: the type of forecast used for the lookahead.
        :param: n_test_episodes: The number of simulated episodes (useful only with noisy forecast).
        """
        super().__init__(env)
        self.grid = None
        self.model = None
        self.control_horizon = control_horizon
        self.simulation_horizon = simulation_horizon
        self.save_data = save_data
        self.path_to_store_experience = path_to_store_experience
        self.forecaster = None
        self.forecast_type = forecast_type
        self.n_test_episodes = n_test_episodes
        self.options_filename = options_filename

    @staticmethod
    def name():
        return "Optimization"

    def train_agent(self):
        pass  # Nothing to train the Optimization agent with

    def reset_agent(self):
        self.forecaster = Forecaster(simulator=self.env.simulator, control_horizon=self.control_horizon,
                                     deviation_factor=0.2)
        self.grid = self.env.simulator.grid

    def simulate_agent(self, agent_options=None):
        for i in range(1, self.n_test_episodes + 1):
            state = self.env.reset()
            self.reset_agent()
            cumulative_reward = 0.0
            done = False
            states = []
            actions = []
            forecasts = []
            while not done:
                self._create_model(self.env.simulator.env_step)
                logger.info("SOLVING: " + str(self.env.simulator.grid_states[-1].date_time))
                low_level_actions = self.get_optimal_action()  # Take the optimized low level actions #

                states.append(self.env.simulator.grid_states[-1])
                actions.append(low_level_actions[0].to_list())
                forecasts.append([self.forecaster.forecasted_consumption, self.forecaster.forecasted_PV_production])

                for j in range(len(low_level_actions)):
                    logger.info("simulating for " + str(self.env.simulator.grid_states[-1].date_time))
                    next_state, reward, done, info = self.env.step(state=state, action=low_level_actions[
                        j])  # Run the simulator in continuous mode (low_level_action)

                cumulative_reward += reward
                state = deepcopy(next_state)
            print('Finished simulation: %d and the reward is: %d.' % (i, cumulative_reward))
            self.env.simulator.store_and_plot(
                folder="results/" + self.name() + "/" + self.env.simulator.case + "/" + self.options_filename + "/" + time_string_for_storing_results(
                    self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime(
                        "%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                    self.env.simulator.case) + "_" + str(i), agent_options=agent_options)

        if self.save_data and self.env.purpose == "Train":
            if not os.path.isdir(self.path_to_store_experience):
                os.makedirs(self.path_to_store_experience)

            details_dict = dict()
            details_dict["train_from_date"] = self.env.simulator.start_date.strftime('%Y-%m-%d %H:%M:%S')
            details_dict["train_to_date"] = self.env.simulator.end_date.strftime('%Y-%m-%d %H:%M:%S')
            details_dict["control_horizon"] = self.control_horizon
            details_dict["simulation_horizon"] = self.simulation_horizon

            with open(self.path_to_store_experience + "/" + self.env.simulator.case + "_optimization_experience_" + str(
                    self.control_horizon) + ".json", "w") as f:
                json.dump(details_dict, f)

            with open(self.path_to_store_experience + "/" + self.env.simulator.case + "_optimization_experience_" + str(
                    self.control_horizon) + ".p",
                      "wb") as fp:  # Pickling
                pickle.dump([states, forecasts, actions], fp)

    def _create_model(self, env_step):
        """
        Define the main elements of the optimization problem
            * the sets
            * the variables
            * the parameters
            * the constraints
            * the objective

        Create and update the model. Return nothing.
        """
        t_build = time.time()
        if self.forecast_type == "exact":
            self.forecaster.exact_forecast(env_step)
        elif self.forecast_type == "noisy":
            self.forecaster.noisy_forecast(env_step)
        self.model = ConcreteModel()
        self._create_sets()
        self._create_parameters()
        self._create_variables()
        self._create_constraints()
        self._create_objective()

        t_build = time.time() - t_build

        logger.debug("Time spent building the mathematical program: %gs" % t_build)

    def _create_sets(self):
        """
        Create the sets of the optimization model.

        Update the model.
        """
        self.model.Periods = RangeSet(
            self.control_horizon)  # The number of periods given to the optimization problem
        self.model.SimulationPeriods = RangeSet(
            self.simulation_horizon)  # The number of  optimized actions steps that will be simulated
        self.model.Batteries = Set(initialize=[s.name for s in self.grid.storages])
        self.model.SteerableGenerators = Set(initialize=[g.name for g in self.grid.generators if g.steerable])
        self.model.NonSteerableGenerators = Set(initialize=[g.name for g in self.grid.generators if not g.steerable])

    def _create_parameters(self):
        """
        Create the parameters of the optimization model.

        Update the model.
        """
        soc = {b.name: k for (b, k) in zip(self.grid.storages, self.env.simulator.grid_states[-1].state_of_charge)}
        total_load = {p: l for (p, l) in zip(self.model.Periods, self.forecaster.forecasted_consumption)}
        total_EPV = {p: g for (p, g) in zip(self.model.Periods, self.forecaster.forecasted_PV_production)}
        n_cycles = {b.name: k for (b, k) in zip(self.grid.storages, self.env.simulator.grid_states[-1].n_cycles)}
        capacity = {b.name: k for (b, k) in zip(self.grid.storages, self.env.simulator.grid_states[-1].capacities)}
        self.model.total_load = Param(self.model.Periods, initialize=total_load)
        self.model.EPV_production = Param(self.model.Periods, initialize=total_EPV)
        self.model.init_soc = Param(self.model.Batteries, initialize=soc)
        self.model.init_n_cycles = Param(self.model.Batteries, initialize=n_cycles)
        self.model.init_capacity = Param(self.model.Batteries, initialize=capacity)

    def _create_variables(self):
        """
        Create the variables of the optimization model.

        Update the model.
        """

        self.model.grid_imp = Var(self.model.Periods, within=NonNegativeReals)  # Auxiliary variable
        self.model.grid_exp = Var(self.model.Periods, within=NonNegativeReals)  # Auxiliary variable
        self.model.production = Var(self.model.Periods, within=NonNegativeReals)  # Auxiliary variable
        self.model.consumption = Var(self.model.Periods, within=NonNegativeReals)  # Auxiliary variable
        self.model.next_soc = Var(self.model.Periods, self.model.Batteries, within=NonNegativeReals)  # State variable
        self.model.charge_power = Var(self.model.Periods, self.model.Batteries,
                                      within=NonNegativeReals)  # Decision variable
        self.model.discharge_power = Var(self.model.Periods, self.model.Batteries,
                                         within=NonNegativeReals)  # Decision variable
        self.model.steer = Var(self.model.Periods, self.model.SteerableGenerators, within=NonNegativeReals,
                               bounds=(0, 1))  # Decision variable
        self.model.bin_battery = Var(self.model.Periods, self.model.Batteries, domain=Binary)  # Auxiliary variable
        self.model.bin_steer = Var(self.model.Periods, self.model.SteerableGenerators, within=Binary,
                                   doc="Gen activation")  # Auxiliary variable
        self.model.next_n_cycles = Var(self.model.Periods, self.model.Batteries,
                                       within=NonNegativeReals)  # State variable
        self.model.next_capacity = Var(self.model.Periods, self.model.Batteries,
                                       within=NonNegativeReals)  # State variable
        self.model.fuel_cost = Var(self.model.Periods, self.model.SteerableGenerators, within=NonNegativeReals)

    def _create_constraints(self):
        """
        Create the constraints of the optimization model.

        Update the model.
        """

        def energy_balance(m, p):
            d = self.grid.period_duration
            # Off-grid assumption: The grid import is lost load and the import export is curtailment 
            return m.grid_imp[p] - m.grid_exp[p] == d * (m.consumption[p] - m.production[p])

        def production_def(m, p):
            rhs = m.EPV_production[p]
            rhs += sum(m.steer[p, g.name] * g.capacity for g in self.grid.generators if g.steerable)
            rhs += sum(m.discharge_power[p, b] for b in self.model.Batteries)
            return m.production[p] == rhs

        def consumption_def(m, p):
            rhs = m.total_load[p]
            rhs += sum(m.charge_power[p, b] for b in self.model.Batteries)
            return m.consumption[p] == rhs

        def soc_evolution(m, p, b):
            assert (len([s for s in self.grid.storages if s.name == b]) == 1)
            storage = [s for s in self.grid.storages if s.name == b][0]
            d = self.grid.period_duration
            charge = d * storage.charge_efficiency * m.charge_power[p, b]
            discharge = (d / storage.discharge_efficiency) * m.discharge_power[p, b]
            if p == 1:
                return m.next_soc[p, b] == m.init_soc[b] + charge - discharge
            else:
                return m.next_soc[p, b] == m.next_soc[p - 1, b] + charge - discharge

        def n_cycles_evolution(m, p, b):
            assert (len([s for s in self.grid.storages if s.name == b]) == 1)
            storage = [s for s in self.grid.storages if s.name == b][0]
            d = self.grid.period_duration
            charge = d * storage.charge_efficiency * m.charge_power[p, b]
            discharge = (d / storage.discharge_efficiency) * m.discharge_power[p, b]
            if p == 1:
                return m.next_n_cycles[p, b] == m.init_n_cycles[b] + (charge + discharge) / (2 * m.init_capacity[
                    b])  # For linearity, current capacity is approximated to the initial capacity of the optimization problem
            else:
                # return  m.next_n_cycles[p, b] == m.next_n_cycles[p - 1, b] + (charge + discharge) * d / (2 * m.next_capacity[p - 1, b]) # real constraint but not linear
                return m.next_n_cycles[p, b] == m.next_n_cycles[p - 1, b] + (charge + discharge) / (2 * m.init_capacity[
                    b])  # For the sake of linearity, current capacity is approximated to the initial capacity of the optimization problem

        def capacity_evolution(m, p, b):
            assert (len([s for s in self.grid.storages if s.name == b]) == 1)
            storage = [s for s in self.grid.storages if s.name == b][0]
            if storage.type() == "DCAStorage":
                return m.next_capacity[p, b] == (
                        storage.initial_capacity * (storage.operating_point[1] - 1) / storage.operating_point[0]) * \
                       m.next_n_cycles[p, b] + storage.initial_capacity
            elif storage.type() == "Storage":
                return m.next_capacity[p, b] == storage.capacity

        def max_charge(m, p, b):
            assert (len([s for s in self.grid.storages if s.name == b]) == 1)
            storage = [s for s in self.grid.storages if s.name == b][0]
            return m.charge_power[p, b] <= storage.max_charge_rate * self.model.bin_battery[p, b]

        def max_discharge(m, p, b):
            assert (len([s for s in self.grid.storages if s.name == b]) == 1)
            storage = [s for s in self.grid.storages if s.name == b][0]
            return m.discharge_power[p, b] <= storage.max_discharge_rate * (1 - m.bin_battery[p, b])

        def max_soc(m, p, b):
            MAX_SOC_MARGIN = 1
            return m.next_soc[p, b] <= m.next_capacity[
                p, b] * MAX_SOC_MARGIN  # FIXME maybe a better idea would be to detect that SOC will be at its max value and to communicate to the simulator it should charge the battery full.

        def min_soc(m, p, b):
            MIN_SOC_MARGIN = 0
            return m.next_soc[p, b] >= m.next_capacity[
                p, b] * MIN_SOC_MARGIN  # FIXME maybe a better idea would be to detect that SOC will be at its min value and to communicate to the simulator it should discharge the battery full.

        def min_stable_gen_low(m, p, g):
            assert (len([gen for gen in self.grid.generators if gen.name == g]) == 1)
            generator = [gen for gen in self.grid.generators if gen.name == g][0]
            return m.steer[p, g] >= generator.min_stable_generation * m.bin_steer[p, g]

        def min_stable_gen_up(m, p, g):
            return m.steer[p, g] <= m.bin_steer[p, g]

        def fuel_cost_1(m, p, g):
            d = self.grid.period_duration
            assert (len([gen for gen in self.grid.generators if gen.name == g]) == 1)
            generator = [gen for gen in self.grid.generators if gen.name == g][0]
            slope = (generator.operating_point_1[1] - generator.operating_point_2[1]) / (
                    generator.operating_point_1[0] - generator.operating_point_2[0])
            intercept = generator.operating_point_2[1] - generator.operating_point_2[0] * slope
            max_cost = (intercept + slope * generator.capacity) * d * generator.diesel_price
            return m.fuel_cost[p, g] <= m.bin_steer[p, g] * max_cost

        def fuel_cost_2(m, p, g):
            d = self.grid.period_duration
            assert (len([gen for gen in self.grid.generators if gen.name == g]) == 1)
            generator = [gen for gen in self.grid.generators if gen.name == g][0]
            slope = (generator.operating_point_1[1] - generator.operating_point_2[1]) / (
                    generator.operating_point_1[0] - generator.operating_point_2[0])
            intercept = generator.operating_point_2[1] - generator.operating_point_2[0] * slope
            cost = (intercept + slope * m.steer[p, g] * generator.capacity) * d * generator.diesel_price
            return m.fuel_cost[p, g] <= cost

        def fuel_cost_3(m, p, g):
            d = self.grid.period_duration
            assert (len([gen for gen in self.grid.generators if gen.name == g]) == 1)
            generator = [gen for gen in self.grid.generators if gen.name == g][0]
            slope = (generator.operating_point_1[1] - generator.operating_point_2[1]) / (
                    generator.operating_point_1[0] - generator.operating_point_2[0])
            intercept = generator.operating_point_2[1] - generator.operating_point_2[0] * slope
            cost = (intercept + slope * m.steer[p, g] * generator.capacity) * d * generator.diesel_price
            max_cost = (intercept + slope * generator.capacity) * d * generator.diesel_price
            return m.fuel_cost[p, g] >= cost - (1 - m.bin_steer[p, g]) * max_cost

        def load_shedding_limits(m, p):
            return m.grid_imp[p] <= m.total_load[p]

        self.model.energy_balance_cstr = Constraint(self.model.Periods, rule=energy_balance)
        self.model.production_def_cstr = Constraint(self.model.Periods, rule=production_def)
        self.model.consumption_def_cstr = Constraint(self.model.Periods, rule=consumption_def)
        self.model.soc_evolution_cstr = Constraint(self.model.Periods, self.model.Batteries, rule=soc_evolution)
        self.model.max_charge_bound = Constraint(self.model.Periods, self.model.Batteries, rule=max_charge)
        self.model.max_discharge_bound = Constraint(self.model.Periods, self.model.Batteries, rule=max_discharge)
        self.model.max_soc_bound = Constraint(self.model.Periods, self.model.Batteries, rule=max_soc)
        self.model.min_soc_bound = Constraint(self.model.Periods, self.model.Batteries, rule=min_soc)
        self.model.min_stable_gen_low = Constraint(self.model.Periods, self.model.SteerableGenerators,
                                                   rule=min_stable_gen_low)
        self.model.min_stable_gen_up = Constraint(self.model.Periods, self.model.SteerableGenerators,
                                                  rule=min_stable_gen_up)
        self.model.n_cycles_evolution_cstr = Constraint(self.model.Periods, self.model.Batteries,
                                                        rule=n_cycles_evolution)
        self.model.capacity_evolution_cstr = Constraint(self.model.Periods, self.model.Batteries,
                                                        rule=capacity_evolution)
        self.model.fuel_cost_1 = Constraint(self.model.Periods, self.model.SteerableGenerators, rule=fuel_cost_1)
        self.model.fuel_cost_2 = Constraint(self.model.Periods, self.model.SteerableGenerators, rule=fuel_cost_2)
        self.model.fuel_cost_3 = Constraint(self.model.Periods, self.model.SteerableGenerators, rule=fuel_cost_3)
        self.model.load_shedding_limits = Constraint(self.model.Periods, rule=load_shedding_limits)

    def _create_objective(self):
        """
        Create the objective function of the optimization model.

        Update the model.
        """

        def total_cost(m):
            # d = self.grid.period_duration
            cost = sum(
                m.grid_imp[p] * self.grid.load_shedding_price + m.grid_exp[p] * self.grid.curtailment_price for p in
                self.model.Periods)
            # cost += sum(sum(m.steer[p,g.name] * g.capacity * d * g.diesel_price/3 for g in self.grid.generators if g.steerable) for p in self.model.Periods) # The fuel cost is assumed to be  1/3 the diesel price, because the fuel cost is a non linear function of the fuel production.
            cost += sum(sum(m.fuel_cost[p, g] for g in self.model.SteerableGenerators) for p in self.model.Periods)
            return cost

        self.model.objFct = Objective(rule=total_cost, sense=minimize)

    def get_optimal_action(self):
        """
        Solve the optimization problem.

        Return a list of GridAction objects containing the optimal charge, discharge and steerable generation levels that will be simulated.
        """
        GridActions = []
        solver = SolverFactory("gurobi")
        results = solver.solve(self.model)
        for p in self.model.SimulationPeriods:
            charge = [value(self.model.charge_power[p, b]) for b in self.model.Batteries]
            discharge = [value(self.model.discharge_power[p, b]) for b in self.model.Batteries]
            generation = {g.name: 0. for g in self.grid.generators if g.steerable}
            for g in self.grid.generators:
                if g.steerable:
                    generation[g.name] = value(self.model.steer[p, g.name]) * g.capacity
            GridActions.append(GridAction(generation, charge, discharge))
        return GridActions


agent_type = OptimizationAgent
