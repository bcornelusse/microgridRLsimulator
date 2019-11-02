from itertools import chain
import pandas as pd
import numpy as np


class Forecaster:

    def __init__(self, simulator, control_horizon, deviation_factor=None):
        """

        :param simulator: Instance of Simulator
        :param control_horizon: The number of forecast steps (includes the current step)
        :param deviation_factor: the std factor used for the noisy forecast
        """
        self.date_range = simulator.date_range
        self.start_date_index = simulator.env_step  # maybe +1
        self.database = simulator.database
        self.grid = simulator.grid
        self.control_horizon = control_horizon  # min(control_horizon, len(
        # self.date_range) - 1 - self.start_date_index)  # -1 because the end date is not part of the problem
        self.forecast_date_range = list(sorted(set(chain(self.date_range,
                                                         pd.date_range(start=self.date_range[-1],
                                                                       periods=self.control_horizon,
                                                                       freq=self.date_range.freq)))))
        self.forecasted_PV_production = None
        self.forecasted_consumption = None
        self.deviation_factor = deviation_factor

    def exact_forecast(self, env_step):
        """
        Make an exact forecast of the future loads and PV production

        Return nothing, fill the forecast lists.
        """

        self.forecasted_PV_production = []
        self.forecasted_consumption = []

        for i in range(self.control_horizon):
            non_flexible_production = 0
            non_flexible_consumption = 0
            for g in self.grid.generators:
                if not g.steerable:
                    time = (env_step + i) * self.grid.period_duration * 60
                    updated_capacity = g.find_capacity(time)
                    non_flexible_production += self.database.get_columns(g.name,
                                                                         self.forecast_date_range[
                                                                             env_step + i]) * (
                                                       updated_capacity / g.initial_capacity)  # Assumption: the capacity update is not taken into account for optimization
            for l in self.grid.loads:
                non_flexible_consumption += self.database.get_columns(l.name,
                                                                      self.forecast_date_range[env_step + i])
            self.forecasted_PV_production.append(non_flexible_production)
            self.forecasted_consumption.append(non_flexible_consumption)

    def noisy_forecast2(self, env_step):
        """
        Make a noisy forecast of the future loads and PV production

        Return nothing, fill the forecast lists.
        """
        self.forecasted_PV_production = []
        self.forecasted_consumption = []

        for i in range(self.control_horizon):
            non_flexible_production = 0
            non_flexible_consumption = 0
            for g in self.grid.generators:
                if not g.steerable:
                    time = (env_step + i) * self.grid.period_duration * 60
                    updated_capacity = g.find_capacity(time)
                    non_flexible_production += self.database.get_columns(g.name,
                                                                         self.forecast_date_range[
                                                                             env_step + i]) * (
                                                       updated_capacity / g.initial_capacity) + \
                                               np.random.normal(scale=self.deviation)
            for l in self.grid.loads:
                non_flexible_consumption += self.database.get_columns(l.name,
                                                                      self.forecast_date_range[env_step + i])
            self.forecasted_PV_production.append(non_flexible_production)
            self.forecasted_consumption.append(non_flexible_consumption)

    def noisy_forecast(self, env_step):
        """
        Make an noise increasing forecast of the future loads and PV production

        Return nothing, fill the forecast lists.
        """
        #This forecast has a variable noise with respect to the forecast step
        self.forecasted_PV_production = []
        self.forecasted_consumption = []
        std_factor = np.linspace(.0, self.deviation_factor, num=self.control_horizon) # increasing from 0 at current step to deviation at final step
        for i in range(self.control_horizon):

            non_flexible_production = 0
            non_flexible_consumption = 0
            for g in self.grid.generators:
                if not g.steerable:
                    time = (env_step + i) * self.grid.period_duration * 60
                    updated_capacity = g.find_capacity(time)
                    non_flexible_production += self.database.get_columns(g.name,
                                                                         self.forecast_date_range[
                                                                             env_step + i]) * (
                                                       updated_capacity / g.initial_capacity)
            for l in self.grid.loads:
                non_flexible_consumption += self.database.get_columns(l.name,
                                                                      self.forecast_date_range[env_step + i])
            self.forecasted_PV_production.append(abs(non_flexible_production + np.random.normal(scale = abs(std_factor[i] * non_flexible_production))))
            # std relative to the exact value => use of a deviation factor increasing from 0% to 20% at last step (can be modified)
            self.forecasted_consumption.append(abs(non_flexible_consumption + np.random.normal(scale = abs(std_factor[i] * non_flexible_consumption))))

    def get_forecast(self):
        return [self.forecasted_consumption, self.forecasted_PV_production]
