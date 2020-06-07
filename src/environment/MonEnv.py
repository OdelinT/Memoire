from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.environments import utils

import unittest
from .place import place


# test from https://github.com/tensorflow/agents/blob/master/tf_agents/environments/tf_environment_test.py
class MonEnv(tf_environment.TFEnvironment):
    def __init__(self, initial_state=0, dtype=tf.int64, scope='TFEnviroment'):
        self.places = []
        for i in range(10):
            self.places.append(place(i))

        self._dtype = dtype
        self._scope = scope
        self._initial_state = tf.cast(initial_state, dtype=self._dtype)
        observation_spec = specs.TensorSpec([1], self._dtype, 'observation')
        time_step_spec = ts.time_step_spec(observation_spec)
        action_spec = specs.BoundedTensorSpec([], tf.int32, minimum=0, maximum=10)
        super(MonEnv, self).__init__(time_step_spec, action_spec)
        self._state = common.create_variable(
            'state', 
            initial_state,
            dtype=self._dtype)
        self.steps = common.create_variable('steps', 0)
        self.resets = common.create_variable('resets', 0)

    def _current_time_step(self):
        step_type, reward, discount = self._state.value()
        return ts.TimeStep(step_type, reward, discount, self._state.value())

    def _reset(self):
        increase_resets = self.resets.assign_add(1)
        with tf.control_dependencies([increase_resets]):
            reset_op = self._state.assign(self._initial_state)
        with tf.control_dependencies([reset_op]):
            time_step = self.current_time_step()
        return time_step

    def _step(self, action):
        action = tf.convert_to_tensor(value=action)
        with tf.control_dependencies(tf.nest.flatten(action)):
            state_assign = self._state.assign_add(1)
        with tf.control_dependencies([state_assign]):
            state_value = self._state.value()
            increase_steps = tf.cond(
                # pred=tf.equal(tf.math.mod(state_value, 3), FIRST),
                true_fn=self.steps.value,
                false_fn=lambda: self.steps.assign_add(1))

        with tf.control_dependencies([increase_steps]):
            return self.current_time_step()

class TFEnvironmentTest(tf.test.TestCase):

    def testResetOp(self):
        tf_env = MonEnv()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(tf_env.reset())
        self.assertEqual(1, self.evaluate(tf_env.resets))
        self.assertEqual(0, self.evaluate(tf_env.steps))

    def testMultipleReset(self):
        tf_env = MonEnv()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(tf_env.reset())
        self.assertEqual(1, self.evaluate(tf_env.resets))
        self.evaluate(tf_env.reset())
        self.assertEqual(2, self.evaluate(tf_env.resets))
        self.evaluate(tf_env.reset())
        self.assertEqual(3, self.evaluate(tf_env.resets))
        self.assertEqual(0, self.evaluate(tf_env.steps))

    def testFirstTimeStep(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        time_step = self.evaluate(time_step)
        # self.assertEqual(FIRST, time_step.step_type)
        self.assertEqual(0.0, time_step.reward)
        self.assertEqual(1.0, time_step.discount)
        self.assertEqual([0], time_step.observation)
        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(0, self.evaluate(tf_env.steps))

    def testFirstStepState(self):
        tf_env = MonEnv()
        tf_env.current_time_step()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(0, self.evaluate(tf_env.steps))

    def testOneStep(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        with tf.control_dependencies([time_step.step_type]):
            action = tf.constant(1)
        next_time_step = tf_env.step(action)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        time_step, next_time_step = self.evaluate([time_step, next_time_step])

        # self.assertEqual(FIRST, time_step.step_type)
        self.assertEqual(0., time_step.reward)
        self.assertEqual(1.0, time_step.discount)
        self.assertEqual([0], time_step.observation)

        # self.assertEqual(MID, next_time_step.step_type)
        self.assertEqual(0., next_time_step.reward)
        self.assertEqual(1.0, next_time_step.discount)
        self.assertEqual([1], next_time_step.observation)

        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(1, self.evaluate(tf_env.steps))

    def testCurrentStep(self):
        if tf.executing_eagerly():
            self.skipTest('b/123881612')
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        with tf.control_dependencies([time_step.step_type]):
            action = tf.constant(1)
        next_time_step = tf_env.step(action)
        self.evaluate(tf.compat.v1.global_variables_initializer())

        time_step_np, next_time_step_np = self.evaluate([time_step, next_time_step])
        # self.assertEqual(FIRST, time_step_np.step_type)
        self.assertEqual(0., time_step_np.reward)
        self.assertEqual(1.0, time_step_np.discount)
        self.assertEqual([0], time_step_np.observation)

        # self.assertEqual(MID, next_time_step_np.step_type)
        self.assertEqual(0., next_time_step_np.reward)
        self.assertEqual(1.0, next_time_step_np.discount)
        self.assertEqual([1], next_time_step_np.observation)

        time_step_np, next_time_step_np = self.evaluate([time_step, next_time_step])
        # self.assertEqual(MID, time_step_np.step_type)
        self.assertEqual(0., time_step_np.reward)
        self.assertEqual(1.0, time_step_np.discount)
        self.assertEqual([1], time_step_np.observation)

        # self.assertEqual(LAST, next_time_step_np.step_type)
        self.assertEqual(1., next_time_step_np.reward)
        self.assertEqual(0.0, next_time_step_np.discount)
        self.assertEqual([2], next_time_step_np.observation)

        time_step_np = self.evaluate(time_step)
        # self.assertEqual(LAST, time_step_np.step_type)
        self.assertEqual(1., time_step_np.reward)
        self.assertEqual(0.0, time_step_np.discount)
        self.assertEqual([2], time_step_np.observation)

        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(2, self.evaluate(tf_env.steps))

    def testTwoStepsDependenceOnTheFirst(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        with tf.control_dependencies([time_step.step_type]):
            action = tf.constant(1)
        time_step = tf_env.step(action)
        with tf.control_dependencies([time_step.step_type]):
            action = tf.constant(2)
        time_step = self.evaluate(tf_env.step(action))
        # self.assertEqual(LAST, time_step.step_type)
        self.assertEqual(1., time_step.reward)
        self.assertEqual(0.0, time_step.discount)
        self.assertEqual([2], time_step.observation)
        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(2, self.evaluate(tf_env.steps))

    def testAutoReset(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        with tf.control_dependencies([time_step.step_type]):
            time_step = tf_env.step(1)
        with tf.control_dependencies([time_step.step_type]):
            time_step = tf_env.step(2)
        with tf.control_dependencies([time_step.step_type]):
            time_step = self.evaluate(tf_env.step(3))
        # self.assertEqual(FIRST, time_step.step_type)
        self.assertEqual(0.0, time_step.reward)
        self.assertEqual(1.0, time_step.discount)
        self.assertEqual([0], time_step.observation)
        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(2, self.evaluate(tf_env.steps))

    def testFirstObservationIsPreservedAfterTwoSteps(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        time_step_np = self.evaluate(time_step)
        self.assertEqual([0], time_step_np.observation)
        time_step = tf_env.step(1)
        with tf.control_dependencies([time_step.step_type]):
            next_time_step = tf_env.step(2)

        observation_np, _ = self.evaluate([time_step.observation, next_time_step])

        self.assertEqual([1], observation_np)

    def testRandomAction(self):
        tf_env = MonEnv()
        time_step = tf_env.current_time_step()
        with tf.control_dependencies([time_step.step_type]):
            action = tf.random.uniform([], minval=0, maxval=10, dtype=tf.int32)
        next_time_step = tf_env.step(action)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        [time_step_np, next_time_step_np] = self.evaluate(
            [time_step, next_time_step])
        self.assertEqual([0], time_step_np.observation)
        self.assertEqual([1], next_time_step_np.observation)
        self.assertEqual(0, self.evaluate(tf_env.resets))
        self.assertEqual(1, self.evaluate(tf_env.steps))

    def testRunEpisode(self):
        tf_env = MonEnv()
        c = lambda t: tf.logical_not(t.is_last())
        body = lambda t: [tf_env.step(t.observation)]

        @common.function
        def run_episode():
            time_step = tf_env.reset()
            return tf.while_loop(cond=c, body=body, loop_vars=[time_step])

        self.evaluate(tf.compat.v1.global_variables_initializer())
        [final_time_step_np] = self.evaluate(run_episode())
        self.assertEqual([2], final_time_step_np.step_type)
        self.assertEqual([2], final_time_step_np.observation)
        self.assertEqual(1, self.evaluate(tf_env.resets))
        self.assertEqual(2, self.evaluate(tf_env.steps))
        # Run another episode.
        [final_time_step_np] = self.evaluate(run_episode())
        self.assertEqual([2], final_time_step_np.step_type)
        self.assertEqual([2], final_time_step_np.observation)
        self.assertEqual(2, self.evaluate(tf_env.resets))
        self.assertEqual(4, self.evaluate(tf_env.steps))