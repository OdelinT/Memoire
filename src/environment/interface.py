from .place import place
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
    def __init__(self, numberOfPlaces = 10, maxSteps = 1000):
        # Environment specific attributes
        self.places = []
        self.maxSteps = maxSteps
        for i in range(numberOfPlaces):
            self.places.append(place(i))

        # PyEnvironment attributes
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.float64, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._reward_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, name='reward')
        self._step_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='step_type')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 1
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))
    
    def _step(self, action):
        estimations = [None] * len(self.places)
        actualDemand = [None] * len(self.places)
        for p in self.places:
            estimations[p.id] = (p.size * action)

        reward = 0
        for p in self.places:
            actualDemand[p.id] = p.getDemand()
            error = estimations[p.id] - actualDemand[p.id]
            reward = error if error<0 else -error
        
        # return ts.termination(np.array([self._state], dtype=np.int32), reward)
        self._state = self._state+1
        if self._state>=self.maxSteps:
            return ts.termination(np.array(actualDemand, dtype=np.int32), reward)
        else:
            return ts.transition(np.array(actualDemand, dtype=np.int32), reward=reward, discount=1.0)
    