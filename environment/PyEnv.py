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
#from .place import place
#from .product import product
import asyncio

tf.compat.v1.enable_v2_behavior()

class PyEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.duration = 30
        self.size = 10

        # Places and products
        # Average size of places: 2000 visits per day
        self.placesSizes = np.random.exponential(size = self.size) * 2000
        
        # Average cost per product: 10
        self.productsCosts = np.random.exponential(size = self.size) * 10
        # Average margin rate: 10%
        self.productsUsualMarginRates = np.random.exponential(size = self.size) / 10
        # Products are on average bought once per hundred of visitors
        self.productsUsualBuyingRates = np.random.exponential(size = self.size) /100
        self.productsUsualPrices = self.productsCosts / (1 - self.productsUsualMarginRates)

        # Specs
        self.initial_observation = np.zeros((self.size,self.size), dtype=np.float32)

        # Action is an array of all the product prices
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
        # TODO parallelize this loop or use numpy to speed up
        for i in range(len(self.placesSizes)):
            # Price elasticity: we'll consider that doubling the price divides the quantity by ten
            quantityLine = np.round((self.placesSizes[i]  * self.productsUsualBuyingRates) * (10 ** ((self.productsUsualPrices - action) / self.productsUsualPrices)))

            marginPerProduct = (action - self.productsCosts) * quantityLine

            reward += marginPerProduct.sum()
            observation[i] = quantityLine
        # convert to numpy array of float32, otherwise not accepted by specs
        observation = np.array(observation, dtype=np.float32)
        if self._state < self.duration:
            self._state += 1
            return ts.transition(observation, reward)
        else:
            return ts.termination(observation, reward)