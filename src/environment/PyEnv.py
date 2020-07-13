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

tf.compat.v1.enable_v2_behavior()

class PyEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.places = []
        for i in range(10):
            self.places.append(place(i))
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
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
        goodValue = np.random.rand() > .5
        if action == goodValue:
            reward = 1
        else:
            reward = -1
        self._state += 1
        if self._state >= 10:
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(np.array([self._state], dtype=np.int32), reward)