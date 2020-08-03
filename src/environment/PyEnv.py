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
import asyncio

tf.compat.v1.enable_v2_behavior()

class PyEnv(py_environment.PyEnvironment):
    def __init__(self, size = 100, duration = 365):
        self.duration = duration
        self.size = size

        self.placesSizes = np.random.exponential(size=100) * 2000
        
        self.productsCosts = np.random.exponential(size = 100) * 10
        self.productsUsualMarginRates = np.random.random(size = 100)
        self.productsUsualMarginRates = np.random.random(size = 100)
        self.productsUsualBuyingRates = np.random.exponential(size=100) /20
        self.productsUsualPrices = self.productsCosts / (1 - self.productsUsualMarginRates)


        self.initial_observation = np.zeros((self.size,self.size), dtype=np.float32)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.size,), dtype=np.float32, minimum=0, maximum=1000, name='action')
        
        self._observation_spec = array_spec.ArraySpec(
            shape = (self.size,self.size),dtype=np.float32,name = 'observation')
        
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
        return ts.restart(self.initial_observation)
    
    def _step(self, action):
        observation = [None] * len(self.placesSizes)
        reward = 0
        # TODO parallelize these loops or use numpy to speed up
        for i in range(len(self.placesSizes)):
            quantityLine = self.placesSizes * ((action / self.productsUsualPrices) * self.productsUsualBuyingRates)

            marginPerProduct = (action / self.productsCosts) * quantityLine

            reward += marginPerProduct.sum()
            observation[i] = quantityLine
        
        

        # convert to numpy array, otherwise not accepted by specs
        observation = np.array(observation)
        if self._state < self.duration:
            self._state += 1
            return ts.transition(observation, reward)
        else:
            return ts.termination(observation, reward)