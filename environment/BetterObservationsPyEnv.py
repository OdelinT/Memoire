from tf_agents.trajectories import time_step as ts
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers

import random
import numpy as np

tf.compat.v1.enable_v2_behavior()

class BetterObservationsPyEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.duration = 30
        self.size = 10
        
        # IMPORTANT
        # Needed to be able to compare different environment's results
        random.seed(0)
        np.random.seed(0)

        # Places and products
        # Average size of places: 2000 visits per day
        self.placeSize = random.random() * 2000

        # Average cost per product: 10
        self.productsCosts = np.random.exponential(size = self.size) * 10
        # Average margin rate: 10%
        self.productsUsualMarginRates = np.random.exponential(size = self.size) / 10
        # Products are on average bought once per hundred of visitors
        self.productsUsualBuyingRates = np.random.exponential(size = self.size) /100
        self.productsUsualPrices = self.productsCosts / (1 - self.productsUsualMarginRates)

        # Price flexibility between 5 and 15
        self.productsPriceFlexibility = np.random.random(size = self.size) * 10 + 5

        # Specs
        self.initial_observation = np.zeros((6,10), dtype=np.float32)
        
        # Action is an array of all the product prices, explained in product cost multiplication
        # This environment doesn't allow to sell at lost to train faster
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.size,), dtype=np.float32, minimum=1, maximum=100, name='action')

        self._observation_spec = array_spec.ArraySpec(
            shape = (6, self.size),dtype=np.float32,name = 'observation')
        
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        #self.__init__()
        return ts.restart(self.initial_observation)
    
    def _step(self, action):
        try:
            if action.shape != self.productsCosts.shape:
                #action shape doesn't match action_spec. It matches observation_spec. Bug or usual for SAC agent?
                #print(action.shape)
                action = np.sum(action, axis=0)
            prices = self.productsCosts * action
            quantities = np.round((self.placeSize  * self.productsUsualBuyingRates) * (self.productsPriceFlexibility ** ((self.productsUsualPrices - prices) / self.productsUsualPrices)))

            marginPerProduct = (prices - self.productsCosts) * quantities

            reward = marginPerProduct.sum()

            observation = np.stack((
                self.productsCosts,
                self.productsUsualMarginRates,
                self.productsUsualBuyingRates,
                self.productsUsualPrices,
                self.productsPriceFlexibility,
                quantities))
            # convert to numpy array of float32, otherwise not accepted by specs
            observation = np.array(observation, dtype=np.float32)

            if self._state < self.duration:
                self._state += 1
                return ts.transition(observation, reward)
            else:
                return ts.termination(observation, reward)
        except:
            1/0