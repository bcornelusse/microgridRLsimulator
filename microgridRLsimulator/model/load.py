from microgridRLsimulator.model.device import Device


class Load(Device):
    def __init__(self, name, capacity):
        """

        :param name: Cf. parent class
        :param capacity: Max rated power of the load.
        """

        super(Load, self).__init__(name)

        self.capacity = capacity
