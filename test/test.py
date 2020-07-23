import unittest
from tf_agents.environments import utils
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from src.environment.PyEnv import PyEnv
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
import tf_agents

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common



num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

class test(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.duration = 10
        # Overriding the default value of the parameters breaks utils.validate_py_environment
        # because it will use the default parameters when sending actions to the environment, regardless the actual parameters
        # self.env = PyEnv(self.size, self.duration)
        self.env = PyEnv()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.env)
    """
    def testValidate(self):
        # doesn't work if we use different size that the default value
        utils.validate_py_environment(self.env, episodes=5)

    def testUselessAction(self):
        action = []
        for i in range(self.env.size):
            action.append(i)
        keep = True
        while keep:
            result = self.env._step(action)
            if False:
                print("Observation: ", result[0])
                print("Reward: ", result[1])
            # 2=index of discount variable in time_step.transition and time_step.termination
            keep = result[2]
    """
    # Doesn't work
    # Doc: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DqnAgent?hl=nl
    # Error: Network only supports action_specs with shape in [(), (1,)])
    def testQNAgent(self):

        fc_layer_params = (100,)

        q_net = q_network.QNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.Variable(0)

        agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        agent.initialize()
    
    # Doesn't work
    # doc: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent?hl=nl
    # Error: Inputs to NormalProjectionNetwork must match the sample_spec.dtype
    def testReinforce(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)

        tf_agent = reinforce_agent.ReinforceAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter)
        tf_agent.initialize()
    
    def testTd3(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
        critic_net = actor_distribution_network.ActorDistributionNetwork(
            self.tf_env.observation_spec(),
            self.tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        actor = tf_agents.agents.Td3Agent(
            time_step_spec = self.tf_env.time_step_spec, 
            action_spec = self.tf_env.action_spec,
            actor_network = actor_net,
            critic_network = critic_net,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer, 
            exploration_noise_std=0.1, critic_network_2=None,
            target_actor_network=None, target_critic_network=None,
            target_critic_network_2=None, target_update_tau=1.0, target_update_period=1,
            actor_update_period=1, dqda_clipping=None, td_errors_loss_fn=None, gamma=1.0,
            reward_scale_factor=1.0, target_policy_noise=0.2, target_policy_noise_clip=0.5,
            gradient_clipping=None, debug_summaries=False, summarize_grads_and_vars=False,
            train_step_counter=None, name=None
        )
