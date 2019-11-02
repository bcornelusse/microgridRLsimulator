"""
The microgridRLsimulator package organizes the test-bench functionalities in subpackages.
"""
from gym.envs.registration import register

register(
    id='microgridRLsimulator-v0',
    entry_point='microgridRLsimulator.gym_wrapper:MicrogridEnv',
)
# __path__ = __import__('pkgutil').extend_path(__path__, __name__)