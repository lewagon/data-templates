"""
Configuration Module.

This module defines a singleton-type configuration class that can be used all across our project. This class can contain any parameter that one may want to change from one simulation run to the other.
"""

import random


class Configuration:
    """
    This configuration class is extremely flexible due to a two-step init process. We only instantiate a single instance of it (at the bottom if this file) so that all modules can import this singleton at load time. (As python always cache module imports, the import actually only happens once). Then, the second initialization happens in main.py and allows the user to input custom parameters of the config class at execution time - and change them as the please during execution.
    """

    def __init__(self):
        """
        Declare types but do not instantiate anything
        """
        self.alpha = 0.2
        self.gamma = 0.98
        self.epsilon = None
        self.rnd_seed = None
        self.agt_type = None

    def init(self, agt_type, **kwargs):
        """
        User-defined configuration init. Mandatory to properly set all configuration parameters.
        """

        # Mandatory arguments go here. In our case it is useless.
        self.agt_type = agt_type

        # We set default values for arguments if we want here
        self.rnd_seed = random.randint(0, 1000)
        self.epsilon = 0.05

        # However, these arguments can be overridden by passing them as keyword arguments in the init method. Hence, passing for instance epsilon=0.1 as a kwarg to the init method will override the default value we just defined.
        self.__dict__.update(kwargs)

        # Once all values are properly set, use them.
        random.seed(self.rnd_seed)


CFG = Configuration()
