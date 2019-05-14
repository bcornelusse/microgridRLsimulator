from microgridRLsimulator.model.device import Device
from microgridRLsimulator.simulate import simulator
from microgridRLsimulator.utils import positive


class Storage(Device):
    def __init__(self, name, params):
        """

        :param name: Cf. parent class
        :param params: dictionary of params, must include a capacity value , a max_charge_rate value,
        a max_discharge_rate value, a charge_efficiency value and a discharge_efficiency value.
        """
        super(Storage, self).__init__(name)

        self.capacity = None
        self.max_charge_rate = None
        self.max_discharge_rate = None
        self.charge_efficiency = 1.0
        self.discharge_efficiency = 1.0
        self.n_cycles = 0
        for k in params.keys():
            if k in self.__dict__.keys():
                self.__setattr__(k, params[k])

        assert (self.capacity is not None)
        assert (self.max_charge_rate is not None)
        assert (self.max_discharge_rate is not None)

    @staticmethod
    def type():
        return "Storage"

    def actual_power(self, charge_action, discharge_action):
        """

        :param charge_action: Charge action from a controller
        :param discharge_action: Discharge action from a controller
        :return: the actual charge and the actual discharge.
        """

        actual_charge = charge_action
        actual_discharge = discharge_action
        # Take care of potential simultaneous charge and discharge.
        if positive(charge_action) and positive(discharge_action):
            net = charge_action - discharge_action
            if net > simulator.TOL_IS_ZERO:
                actual_charge = net
                actual_discharge = 0.0
            elif net < -simulator.TOL_IS_ZERO:
                actual_charge = 0
                actual_discharge = -net
        return actual_charge, actual_discharge

    def update_cycles(self, throughput, deltat):
        """

        Update the storage number of cycles.

        :param throughput: power charged or discharged, in absolute value [kW]
        :param deltat: period duration [h]
        :return: Nothing, updates number of cycles.
        """
        self.n_cycles += throughput * deltat / (2 * self.capacity)

    def simulate(self, initial_soc, charge_action, discharge_action, deltat):
        """

        :param initial_soc: initial state of charge of the battery [kWh]
        :param charge_action: Charge action from a controller [kW]
        :param discharge_action: Discharge action from a controller [kW]
        :param deltat : Period duration [h]
        :return: the next state of charge, the actual charge and the actual discharge.
        """

        next_soc = initial_soc
        actual_charge, actual_discharge = self.actual_power(charge_action, discharge_action)

        if positive(actual_charge):
            planned_evolution = initial_soc + actual_charge * deltat * self.charge_efficiency # TODO check action is an energy: action is a power 
            next_soc = min(self.capacity, planned_evolution)                                  # period duration used to modify it in an energy
            actual_charge = (next_soc - initial_soc) / (self.charge_efficiency * deltat)
        elif positive(actual_discharge):
            planned_evolution = initial_soc - actual_discharge * deltat / self.discharge_efficiency # same for discharge
            next_soc = max(0, planned_evolution)
            actual_discharge = (initial_soc - next_soc) * self.discharge_efficiency / deltat

        return next_soc, actual_charge, actual_discharge