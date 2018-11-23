from .load import Load
from .storage import Storage
from .generator import Generator


class Grid:
    def __init__(self, data):
        """
        A microgridRLsimulator is represented by its devices which are either loads, generators or storage
        devices, and additional information such as prices.
        The period duration of the simulation is also stored at this level, although
        it is more part of the configuration of the simulation.

        :param data: A json type dictionary containing a description of the microgridRLsimulator.
        """
        self.loads = [Load(l["name"], l["capacity"]) for l in data["loads"]]
        self.generators = [Generator(g["name"], g) for g in data["generators"]]
        self.storages = [Storage(s["name"], s) for s in data["storages"]]

        self.period_duration = data["period_duration"]  # TODO period_duration to a config file?

        self.curtailement_price = data["curtailement_price"]
        self.load_shedding_price = data["load_shedding_price"]

    @property
    def base_purchase_price(self):
        return self._base_purchase_price

    @base_purchase_price.setter
    def base_purchase_price(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self._base_purchase_price = float(value)

    @property
    def peak_price(self):
        return self._peak_price

    @peak_price.setter
    def peak_price(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self._peak_price = float(value)

    @property
    def period_duration(self):
        return self._period_duration

    @period_duration.setter
    def period_duration(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self._period_duration = float(value)

    @property
    def price_margin(self):
        return self._price_margin

    @price_margin.setter
    def price_margin(self, value):
        assert isinstance(value, int) or isinstance(value, float)
        self._price_margin = float(value)

    def purchase_price(self, energy_prices):
        """

        :param energy_prices: A list of energy prices (i.e. a time series), in EUR/MWh
        :return: The actual purchase price taking into account all components, in EUR/kWh
        """
        return [self.base_purchase_price + p * (1 + self.price_margin) * 1e-3 for p in
                energy_prices]

    def sale_price(self, energy_prices):
        """

        :param energy_prices: A list of energy prices (i.e. a time series), in EUR/MWh
        :return: The actual sale price taking into account all components, in EUR/kWh
        """
        return [p * (1 - self.price_margin) * 1e-3 for p in energy_prices]

    def get_non_flexible_device_names(self):
        """

        :return: The list of names of all non-flexible loads and generators for which there must be an entry in the data history
        """
        names = [d.name for d in self.loads]
        for d in self.generators:
            if not d.steerable:
                names.append(d.name)

        return names
