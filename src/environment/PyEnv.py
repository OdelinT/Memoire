import numpy as np
from tf_agents.trajectories import time_step as ts
# import abc
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
#from tf_agents.environments import suite_gym
from .place import place
from .product import product

tf.compat.v1.enable_v2_behavior()

class PyEnv(py_environment.PyEnvironment):
    def __init__(self, size = 100, duration = 365):
        self.duration = duration
        self.places = []
        for i in range(size):
            self.places.append(place())
        self.products = []
        for i in range(size):
            self.products.append(product())
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(size,), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(size,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        #self._state = 0
        #self._episode_ended = False
        self.__init__()
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        observation = []
        reward = 0
        for i in range(len(self.places)):
            observation.append([])
            for j in range(len(self.products)):
                price = action[j]
                quantity = self.places[i].getDemand(self.products[j], price)
                margin = self.products[j].getMargin(price, quantity)
                observation[i].append((quantity, margin))
                reward += margin
        if self._state < self.duration:
            self._state += 1
            # return ts.transition(observation, reward)
            return observation, reward, True
        else:
            # return ts.termination(observation, reward)
            return observation, reward, False