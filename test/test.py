import unittest
from tf_agents.environments import utils
import tensorflow as tf
from tf_agents.networks import q_network, normal_projection_network, actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
from src.environment.PyEnv import PyEnv
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
import tf_agents
from datetime import datetime
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent

import numpy as np

# Test cases are commented and decommented according to what I was studying during commit.

class test(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.duration = 10
        # Overriding the default value of the parameters breaks utils.validate_py_environment
        # because it will use the default parameters when sending actions to the environment, regardless the actual parameters
        # self.env = PyEnv(self.size, self.duration)
        self.env = PyEnv()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.env)
        self.train_env = self.tf_env

    def testMyValidate(self):
        self.env._reset()

        for value in self.env.placesSizes:
            if value < 0:
                raise Exception()
        for value in self.env.productsCosts:
            if value < 0:
                raise Exception()
        for value in self.env.productsUsualMarginRates:
            if value < 0:
                raise Exception()
            if value > 1:
                raise Exception()
        for value in self.env.productsUsualBuyingRates:
            if value < 0:
                raise Exception()
        for i in range(len(self.env.productsUsualPrices)):
            if self.env.productsUsualPrices[i] < self.env.productsCosts[i]:
                raise Exception()

        observation = self.env._step(self.env.productsCosts)
        print("Product costs: ", observation)
        costReward = observation.reward
        if observation.reward != 0:
            print("Error: If we sell at product cost, there should be no margin")

        observation = self.env._step(self.env.productsCosts * 2)
        print("Product costs *2: ", observation)
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            print("Error: If we sell at more than product cost, there should be a margin")

        observation = self.env._step(self.env.productsCosts * 10)
        print("Product costs *10: ", observation)
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            print("Error: If we sell at more than product cost, there should be a margin")

        observation = self.env._step(self.env.productsUsualPrices)
        print("Usual buying price: ", observation)
        usualReward = observation.reward

        observation = self.env._step(self.env.productsUsualPrices * 2)
        print("Usual buying price *2: ", observation)
        print("Compared to usual: ", observation.reward / usualReward)

        observation = self.env._step(self.env.productsUsualPrices * 4)
        print("Usual buying price *4: ", observation)
        print("Compared to usual: ", observation.reward / usualReward)

        observation = self.env._step(self.env.productsUsualPrices * 10)
        print("Usual buying price *10: ", observation)
        print("Compared to usual: ", observation.reward / usualReward)

        observation = self.env._step(np.zeros(100))
        print("0: ", observation)
        if observation.reward >= 0.:
            print("Error: If prices are 0, we should sell and have a deficit")

        observation = self.env._step(np.zeros(100) + 1)
        print("1: ", observation)
        if observation.reward >= 0:
            print("Error: If prices are 1 and average cost 10, we should sell and have a deficit")

        observation = self.env._step(np.zeros(100) + 1000)
        print("1000: ", observation)
        if observation.reward > usualReward:
            print("Error: An unusual price such as 1000 shouldn't cause a best result than the usual price. Error ratio: ", observation.reward / usualReward)
        
    """
    def testValidate(self):
        utils.validate_py_environment(self.env, episodes=5)
    """

    """
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

    """
    # Doesn't work
    # Doc: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DqnAgent?hl=nl
    # Error: Network only supports action_specs with shape in [(), (1,)])
    def testQNAgent(self):
        
        num_iterations = 250 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}

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
    """
    """
    # Doesn't work - not supposed to
    # doc: https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent?hl=nl
    # Error: Inputs to NormalProjectionNetwork must match the sample_spec.dtype
    def testReinforceNormalProjectionNet(self):
        # actor_net = actor_distribution_network.ActorDistributionNetwork(
        #     self.tf_env.observation_spec(),
        #     self.tf_env.action_spec(),
        #     fc_layer_params=fc_layer_params)
        
        num_iterations = 250 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}

        actor_net = normal_projection_network.NormalProjectionNetwork(
            self.tf_env.observation_spec())
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
    """
    """
    # Works!
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def testReinforceActorDistributionNet(self):
        print("Starting testReinforceActorDistributionNet")
        num_iterations = 250 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)

        tf_agent = reinforce_agent.ReinforceAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter)
        tf_agent.initialize()

        collect_policy = tf_agent.collect_policy

        print(self.compute_avg_return(self.train_env, collect_policy))
    """
    
    """
    def testReinforceCriticNet(self):
        num_iterations = 250 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)

        tf_agent = reinforce_agent.ReinforceAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter)
        tf_agent.initialize()
    """
    """
    def testTd3(self):
        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}
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
            train_step_counter=None, name=None)
        actor.initialize()
    """
    """
    def normal_projection_net(self, action_spec, init_means_output_factor=0.1):
        return normal_projection_network.NormalProjectionNetwork(
            action_spec,
            mean_transform=None,
            state_dependent_std=True,
            init_means_output_factor=init_means_output_factor,
            std_transform=sac_agent.std_clip_transform,
            scale_distribution=True)
    """
    def normal_projection_net(self, action_spec,init_means_output_factor=0.1):
        return normal_projection_network.NormalProjectionNetwork(
            action_spec,
            mean_transform=None,
            state_dependent_std=True,
            init_means_output_factor=init_means_output_factor,
            std_transform=sac_agent.std_clip_transform,
            scale_distribution=True)

    def compute_avg_return(self, environment, policy, num_episodes=5):
        total_return = 0.0
        print("Starting to compute")
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            print(datetime.now(), " Episode return : ", episode_return)
            total_return += episode_return
        return total_return / num_episodes
        # avg_return = total_return / num_episodes
        # return avg_return.numpy()[0]
    """
    # Works!
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb
    def testSAC(self):
        print("Starting testSAC")
        # use "num_iterations = 1e6" for better results,
        # 1e5 is just so this doesn't take too long. 
        num_iterations = 100000 # @param {type:"integer"}

        initial_collect_steps = 10000 # @param {type:"integer"} 
        collect_steps_per_iteration = 1 # @param {type:"integer"}
        replay_buffer_capacity = 1000000 # @param {type:"integer"}

        batch_size = 256 # @param {type:"integer"}

        critic_learning_rate = 3e-4 # @param {type:"number"}
        actor_learning_rate = 3e-4 # @param {type:"number"}
        alpha_learning_rate = 3e-4 # @param {type:"number"}
        target_update_tau = 0.005 # @param {type:"number"}
        target_update_period = 1 # @param {type:"number"}
        gamma = 0.99 # @param {type:"number"}
        reward_scale_factor = 1.0 # @param {type:"number"}
        gradient_clipping = None # @param

        actor_fc_layer_params = (256, 256)
        critic_joint_fc_layer_params = (256, 256)

        log_interval = 5000 # @param {type:"integer"}

        # num_eval_episodes = 30 # @param {type:"integer"}
        eval_interval = 10000 # @param {type:"integer"}

        observation_spec = self.train_env.observation_spec()
        action_spec = self.train_env.action_spec()

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=self.normal_projection_net)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf_agent = sac_agent.SacAgent(
            self.train_env.time_step_spec(),
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            train_step_counter=global_step)
        tf_agent.initialize()

        collect_policy = tf_agent.collect_policy

        print(self.compute_avg_return(self.train_env, collect_policy))
    """