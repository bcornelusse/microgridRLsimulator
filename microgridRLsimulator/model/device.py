from abc import ABCMeta
class Device(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Base class for all devices.

        :param name: Name of the devices, used as a reference for access to realized data or forecasts
        """
        self.name = name