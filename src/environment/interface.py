from .simulator import simulator
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
tf.compat.v1.enable_v2_behavior()

class interface(py_environment.PyEnvironment):
    def __init__(self):
        self.simulator = simulator()
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
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        places = self.simulator.allPlaces()
        estimations = [None] * len(places)
        for p in places:
            estimations[p.id] = p.size

        reward = 0
        for p in places:
            error = estimations[p.id] - self.simulator.getDemand(p.id)
            reward = error if error<0 else -error
        
        return ts.termination(np.array([self._state], dtype=np.int32), reward)
    