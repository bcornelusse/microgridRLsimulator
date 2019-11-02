from .device import Device


class Generator(Device):

    def __init__(self, name, params):
        """
        
        :param name: Cf. parent class
        :param params: dictionary of params, must include a capacity value , a steerable flag, and a min_stable_generation value
        """

        super(Generator, self).__init__(name)

        self.capacity = None
        self.steerable = False
        self.min_stable_generation = None
        self.diesel_price = None  # €/l
        self.initial_capacity = params["capacity"]
        self.operating_point = None # EPV


        # In oder to determine the efficiency we need two operating points
        # It is considered that the fuel curve is linear (HOMER)
        self.operating_point_1 = None  # Diesel
        self.operating_point_2 = None  # Diesel

        for k in params.keys():
            if k in self.__dict__.keys():
                self.__setattr__(k, params[k])

        assert (self.capacity is not None)

    def update_capacity(self, time):
        """

        Decrease the generator capacity over time using a linear function.

        :return: Nothing, updates the generator capacity.
        """
        self.capacity = self.initial_capacity * (self.operating_point[1] - 1) * time / (24 * 60 * self.operating_point[0]) + self.initial_capacity

    def find_capacity(self, time):
        """

        Calculate the generator capacity at the next step.

        :return: The capacity without updating it. Useful for lookahead.
        """
        return (self.initial_capacity * (self.operating_point[1] - 1) * time / (24 * 60 * self.operating_point[0]) + self.initial_capacity)
        

    def simulate_generator(self, production, simulation_resolution):
        TOL_IS_ZERO = 1e-4

        diesel_lhv = 43.2  # MJ/kg
        diesel_density = 820  # kg/l
        diesel_lhv_kWh = (diesel_lhv * diesel_density) / 3600  # kWh/l

        min_stable_generation = self.capacity * self.min_stable_generation

        slope = (self.operating_point_1[1] - self.operating_point_2[1]) / (self.operating_point_1[0] - self.operating_point_2[0])
        intercept = self.operating_point_2[1] - self.operating_point_2[0] * slope
        F_0 = intercept / self.capacity
        F_1 = slope

        diesel_consumption_l = 0

        if TOL_IS_ZERO < production < min_stable_generation: # TODO This is already verified in construct action, check if this should be removed. Though in continuous action space, construct action method is not used
            genset_generation = min_stable_generation
        elif production > self.capacity:
            genset_generation = self.capacity
        else:
            genset_generation = production

        if genset_generation > TOL_IS_ZERO:
            v_dot_diesel = F_0 * self.capacity + F_1 * genset_generation
            genset_efficiency = genset_generation / (v_dot_diesel * diesel_lhv_kWh)

            diesel_consumption_l = (genset_generation / (diesel_lhv_kWh * genset_efficiency)) * simulation_resolution

        total_cost = diesel_consumption_l * self.diesel_price

        return genset_generation, total_cost
