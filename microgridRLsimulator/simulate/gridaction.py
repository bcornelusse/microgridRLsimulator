# -*- coding: utf-8 -*-


class GridAction():
    def __init__(self, conventional_generation, charge, discharge):
        """
        Action taken by the agent.

        Each action is defined per device, then per period of the optimization horizon.
        Each member is defined as a list or as nested lists.

        :param conventional_generation: Genset generation [kW]
        :param charge: Action to charge storage devices [kW]
        :param discharge: Action to discharge storage devices [kW]
        """

        self.conventional_generation = conventional_generation
        self.charge = charge
        self.discharge = discharge

    def to_json(self):
        return self.__dict__
