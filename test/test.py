#region imports
from environment.BasePyEnv import BasePyEnv
from environment.AllowDeficitCostPyEnv import AllowDeficitCostPyEnv

import unittest
import copy
from tf_agents.environments import utils
import tensorflow as tf
from tf_agents.networks import q_network, normal_projection_network, actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
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
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
import coloredlogs, logging
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
import pickle


from tf_agents.networks import value_network

import numpy as np
#endregion

# Test cases are commented and decommented according to what I was studying during commit.

class test(unittest.TestCase):
    def setUp(self):
        coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')
        self.base_env = BasePyEnv()
        self.base_env2 = copy.deepcopy(self.base_env)
        self.base_env3 = BasePyEnv()
        self.base_tf_env = tf_py_environment.TFPyEnvironment(self.base_env)
        self.base_train_env = tf_py_environment.TFPyEnvironment(self.base_env2)
        self.base_eval_env = tf_py_environment.TFPyEnvironment(self.base_env3)

        self.AllowDeficit_env = AllowDeficitCostPyEnv()
        self.AllowDeficit_env2 = AllowDeficitCostPyEnv()
        self.AllowDeficit_env3 = AllowDeficitCostPyEnv()
        self.AllowDeficit_tf_env = tf_py_environment.TFPyEnvironment(self.AllowDeficit_env)
        self.AllowDeficit_train_env = tf_py_environment.TFPyEnvironment(self.AllowDeficit_env2)
        self.AllowDeficit_eval_env = tf_py_environment.TFPyEnvironment(self.AllowDeficit_env3)
    
    """ 15/08 : bug
    def testBaseEnvParametersCheck(self):
        self.base_env._reset()
        size = self.base_env.size
        #region Test environment parameters generation
        logging.info("Test environment parameters generation")
        for value in self.base_env.productsCosts:
            if value < 0:
                logging.error("At least one product cost < 0")
        for value in self.base_env.productsUsualMarginRates:
            if value < 0:
                logging.error("At least one margin rate < 0")
            if value > 1:
                logging.error("At least one margin rate > 1")
        for value in self.base_env.productsUsualBuyingRates:
            if value <= 0:
                logging.error("At least one buying rate <= 0")
        for i in range(len(self.base_env.productsUsualPrices)):
            if self.base_env.productsUsualPrices[i] < self.base_env.productsCosts[i]:
                logging.error("At least one usual price smaller than product cost")
        #endregion

        #region Test environment credibility with different prices
        logging.info("Test environment credibility with different prices")
        observation = self.base_env._step(self.base_env.productsCosts)
        logging.info(f"Product costs: {observation}")
        costReward = observation.reward
        if observation.reward != 0:
            logging.error("Error: If we sell at product cost, there should be no margin")

        observation = self.base_env._step(self.base_env.productsCosts * 2)
        logging.info(f"Product costs *2: {observation}")
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            logging.error("Error: If we sell at more than product cost, there should be a margin")

        observation = self.base_env._step(self.base_env.productsCosts * 10)
        logging.info(f"Product costs *10: {observation}")
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            logging.error("Error: If we sell at more than product cost, there should be a margin")

        observation = self.base_env._step(self.base_env.productsUsualPrices)
        logging.info(f"Usual buying price: {observation}")
        usualReward = observation.reward

        observation = self.base_env._step(self.base_env.productsUsualPrices * 2)
        logging.info(f"Usual buying price *2: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.base_env._step(self.base_env.productsUsualPrices * 4)
        logging.info(f"Usual buying price *4: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.base_env._step(self.base_env.productsUsualPrices * 10)
        logging.info(f"Usual buying price *10: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.base_env._step(np.zeros(size))
        logging.info(f"0: {observation}")
        if observation.reward >= 0.:
            logging.error("Error: If prices are 0, we should sell and have a deficit")

        observation = self.base_env._step(np.zeros(size) + 1)
        logging.info(f"1: {observation}")
        if observation.reward >= 0:
            logging.error("Error: If prices are 1 and average cost 10, we should sell and have a deficit")

        observation = self.base_env._step(np.zeros(size) + 1000)
        logging.info(f"1000: {observation}")
        if observation.reward > usualReward:
            logging.error(f"Error: An unusual price such as 1000 shouldn't cause a best result than the usual price. Error ratio: {observation.reward / usualReward}")
        #endregion
    """
    
    def testBaseEnvValidate(self):
        utils.validate_py_environment(self.base_env, episodes=5)
    
    def testEnvironmentSeedForParametersGeneration(self):
        duration = self.base_env.duration
        size = self.base_env.size
        placeSize = self.base_env.placeSize
        productsCosts = self.base_env.productsCosts
        productsUsualMarginRates = self.base_env.productsUsualMarginRates
        productsUsualBuyingRates = self.base_env.productsUsualBuyingRates
        productsUsualPrices = self.base_env.productsUsualPrices
        #i = 0
        for environment in [
            self.base_env,
            self.base_env2,
            self.base_env3,
            self.AllowDeficit_env,
            self.AllowDeficit_env2,
            self.AllowDeficit_env3
        ]:
            #i += 1
            #logging.info(i)
            self.assertEqual(duration, environment.duration)
            self.assertEqual(size, environment.size)
            self.assertEqual(placeSize, environment.placeSize)
            self.assertEqual(productsCosts.all(), environment.productsCosts.all())
            self.assertEqual(productsUsualMarginRates.all(), environment.productsUsualMarginRates.all())
            self.assertEqual(productsUsualBuyingRates.all(), environment.productsUsualBuyingRates.all())
            self.assertEqual(productsUsualPrices.all(), environment.productsUsualPrices.all())
    
    #region Agent initialization methods

    # No error but reward=0 almost every time
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def Reinforce(self):
        #region Agent initialization
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.base_train_env.observation_spec(),
            self.base_train_env.action_spec(),
            fc_layer_params=self.fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_step_counter = tf.compat.v2.Variable(0)

        tf_agent = reinforce_agent.ReinforceAgent(
            self.base_train_env.time_step_spec(),
            self.base_train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter)
        tf_agent.initialize()
        #endregion
        return tf_agent
    
    # Error: One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'.
    def TD3(self):
        #region Agent initialization
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.base_tf_env.observation_spec(),
            self.base_tf_env.action_spec(),
            fc_layer_params=self.fc_layer_params)
        critic_net = actor_distribution_network.ActorDistributionNetwork(
            self.base_tf_env.observation_spec(),
            self.base_tf_env.action_spec(),
            fc_layer_params=self.fc_layer_params)
        
        actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        tf_agent = tf_agents.agents.Td3Agent(
            time_step_spec = self.base_tf_env.time_step_spec(), 
            action_spec = self.base_tf_env.action_spec(),
            actor_network = actor_net,
            critic_network = critic_net,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer)
        tf_agent.initialize()
        #endregion
        return tf_agent

    # Error: Network only supports action_specs with shape in [(), (1,)])
    # Doesn't fit what is in the doc example
    # https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
    def DQN(self):
        #region DQN params from doc
        q_net = q_network.QNetwork(
            self.base_train_env.observation_spec(),
            self.base_train_env.action_spec(),
            fc_layer_params=self.fc_layer_params)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        train_step_counter = tf.Variable(0)

        tf_agent = dqn_agent.DqnAgent(
            self.base_train_env.time_step_spec(),
            self.base_train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        tf_agent.initialize()
        #endregion
        return tf_agent

    def SAC(self):
        #region Agent initialization
        observation_spec = self.base_train_env.observation_spec()
        action_spec = self.base_train_env.action_spec()

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=self.critic_joint_fc_layer_params)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=self.actor_fc_layer_params,
            continuous_projection_net=self.normal_projection_net)

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf_agent = sac_agent.SacAgent(
            self.base_train_env.time_step_spec(),
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.alpha_learning_rate),
            target_update_tau=self.target_update_tau,
            target_update_period=self.target_update_period,
            td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=self.gamma,
            reward_scale_factor=self.reward_scale_factor,
            gradient_clipping=self.gradient_clipping,
            train_step_counter=global_step)
        tf_agent.initialize()
        #endregion

        return tf_agent
    
    def BehavioralCloning(self):
        #region Agent initialization
        observation_spec = self.base_train_env.observation_spec()
        action_spec = self.base_train_env.action_spec()
        time_step_spec = self.base_train_env.time_step_spec()
        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=self.critic_joint_fc_layer_params)
        tf_agent = behavioral_cloning_agent.BehavioralCloningAgent(
            time_step_spec, 
            action_spec, 
            cloning_network = critic_net, 
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.actor_learning_rate))
        #endregion
        tf_agent.initialize()
        return tf_agent

    #https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DdpgAgent
    def DDPG(self):
        #region Agent initialization
        observation_spec = self.base_train_env.observation_spec()
        action_spec = self.base_train_env.action_spec()

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=self.critic_joint_fc_layer_params)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=self.actor_fc_layer_params,
            continuous_projection_net=self.normal_projection_net)

        tf_agent = ddpg_agent.DdpgAgent(
            self.base_train_env.time_step_spec(),
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.actor_learning_rate))
        tf_agent.initialize()
        #endregion
        return tf_agent

    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/PPOAgent
    def PPO(self):
        #region Initialize agent
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.base_train_env.observation_spec(),
            self.base_train_env.action_spec(),
            fc_layer_params=self.actor_fc_layer_params)
        value_net = value_network.ValueNetwork(
            self.base_train_env.observation_spec())
        tf_agent = ppo_agent.PPOAgent(
            self.base_train_env.time_step_spec(),
            self.base_train_env.action_spec(),
            optimizer = optimizer,
            actor_net = actor_net,
            value_net = value_net)

        tf_agent.initialize()
        #endregion
        return tf_agent

    #endregion
    
    
    def BaseTest(self):
        #region Hyperparameters from the example of the documentation
        # use "num_iterations = 1e6" for better results,
        # 1e5 is just so this doesn't take too long. 
        self.num_iterations = 1000
        self.log_interval = self.num_iterations / 5
        self.eval_interval = self.num_iterations / 5
        self.num_eval_episodes = 100

        self.collect_steps_per_iteration = 10
        self.initial_collect_steps = self.collect_steps_per_iteration
        self.replay_buffer_capacity = self.num_iterations

        self.batch_size = 256 

        self.learning_rate = 3e-3
        self.critic_learning_rate = self.learning_rate
        self.actor_learning_rate = self.learning_rate
        self.alpha_learning_rate = self.learning_rate
        self.target_update_tau = 0.05 
        self.target_update_period = 1 
        self.gamma = 0.99 
        self.reward_scale_factor = 1.0 
        self.gradient_clipping = None # @param

        self.fc_layer_params = (256, 256)
        self.actor_fc_layer_params = self.fc_layer_params
        self.critic_joint_fc_layer_params = self.fc_layer_params
        #endregion

        training=3
        algo='SAC'

        #for algo in ["SAC", "PPO", "TD3", "DQN", "Reinforce", "DDPG", "BehavioralCloning"]:
        #    for training in [2, 3]:

        logging.info('----------------------------------')
        logging.info(f'Starting to test {algo} with training {training}')
        logging.info('----------------------------------')
        try:
            #region Agent selection
            if algo == 'SAC':
                tf_agent = self.SAC()
            elif algo == 'DQN':
                tf_agent = self.DQN()
            elif algo == 'TD3':
                tf_agent = self.TD3()
            elif algo == 'Reinforce':
                tf_agent = self.Reinforce()
            elif algo == 'PPO':
                tf_agent = self.PPO()
            elif algo == 'DDPG':
                tf_agent = self.DDPG()
            elif algo == 'BehavioralCloning':
                tf_agent = self.BehavioralCloning()
            else:
                raise(f"No algorithm matches {algo}")
            #endregion

            
            collect_policy = tf_agent.collect_policy
            random_policy  = random_tf_policy.RandomTFPolicy(
                self.base_train_env.time_step_spec(),
                self.base_train_env.action_spec())
            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=self.base_train_env.batch_size,
                max_length=self.replay_buffer_capacity)
            
            #region Agent training
            logging.info(f'Starting agent training over {self.num_iterations} steps')
            if training==1:
                #DynamicStepDriver takes too long time
                self.trainAvecDynamicStepDriver(tf_agent=tf_agent, collect_policy=collect_policy, replay_buffer=replay_buffer, initial_collect_steps=self.initial_collect_steps, num_iterations=self.num_iterations, num_eval_episodes=self.num_eval_episodes, eval_interval=self.eval_interval, log_interval=self.log_interval)
            elif training == 2:
                # Error: One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'
                self.trainAvecJusteReplayBuffer(tf_agent=tf_agent, collect_policy=collect_policy, replay_buffer=replay_buffer, initial_collect_steps=self.initial_collect_steps, num_iterations=self.num_iterations, num_eval_episodes=self.num_eval_episodes, eval_interval=self.eval_interval, log_interval=self.log_interval)
            elif training == 3:
                #Has interesting results. First improves a lot, then regresses
                self.train3(tf_agent=tf_agent, collect_policy=collect_policy, replay_buffer=replay_buffer, initial_collect_steps=self.initial_collect_steps, num_iterations=self.num_iterations, num_eval_episodes=self.num_eval_episodes, eval_interval=self.eval_interval, log_interval=self.log_interval)
            elif training == 4:
                #Has interesting results. First improves a lot, then regresses
                self.AlwaysGreedy(tf_agent=tf_agent, collect_policy=collect_policy, replay_buffer=replay_buffer, initial_collect_steps=self.initial_collect_steps, num_iterations=self.num_iterations, num_eval_episodes=self.num_eval_episodes, eval_interval=self.eval_interval, log_interval=self.log_interval)
            else:
                logging.error(f"No training match {training}")
            logging.info('Agent training finished')
            #endregion

            #region Agent training results
            greedy = greedy_policy.GreedyPolicy(tf_agent.policy)
            logging.info('Test agent result')
            self.compute_avg_return(self.base_eval_env, greedy, self.num_eval_episodes, display=False)
            #endregion

            data = replay_buffer.gather_all()

            logging.info('Writing results in a file')
            with open('result.dump', 'wb') as file:
                pickle.dump(data, file)
        except Exception as e:
            logging.error(e)
    
    def testCompareOverParametrized(self):
        #region Hyperparameters from the example of the documentation
        # use "num_iterations = 1e6" for better results,
        # 1e5 is just so this doesn't take too long. 
        self.num_iterations = 10000
        self.log_interval = self.num_iterations / 5
        self.eval_interval = self.num_iterations / 5
        self.num_eval_episodes = 100

        self.collect_steps_per_iteration = 10
        self.initial_collect_steps = self.collect_steps_per_iteration
        self.replay_buffer_capacity = self.num_iterations

        self.batch_size = 256 

        self.learning_rate = 3e-3
        self.critic_learning_rate = self.learning_rate
        self.actor_learning_rate = self.learning_rate
        self.alpha_learning_rate = self.learning_rate
        self.target_update_tau = 0.05 
        self.target_update_period = 1 
        self.gamma = 0.99 
        self.reward_scale_factor = 1.0 
        self.gradient_clipping = None # @param

        self.fc_layer_params = (256, 256)
        self.actor_fc_layer_params = self.fc_layer_params
        self.critic_joint_fc_layer_params = self.fc_layer_params
        #endregion

        training=3
        algo='SAC'

        for env in [
            {
                "train" : self.base_train_env, 
                "eval" : self.base_eval_env, 
                "name": "Environment that don't allow to sell at lost"
            }, {
                "train" : self.AllowDeficit_train_env, 
                "eval" : self.AllowDeficit_eval_env, 
                "name": "Environment that allows to sell at lost"
            }
        ]:
            logging.info('----------------------------------')
            logging.info(f'Starting to test {env["name"]}')
            logging.info('----------------------------------')
            try:
                tf_agent = self.SAC()
                
                collect_policy = tf_agent.collect_policy
                random_policy  = random_tf_policy.RandomTFPolicy(
                    env["train"].time_step_spec(),
                    env["train"].action_spec())
                replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    data_spec=tf_agent.collect_data_spec,
                    batch_size=env["train"].batch_size,
                    max_length=self.replay_buffer_capacity)
                
                #region Agent training
                logging.info(f'Starting agent training over {self.num_iterations} steps')
                self.train3(tf_agent=tf_agent, collect_policy=collect_policy, replay_buffer=replay_buffer, initial_collect_steps=self.initial_collect_steps, num_iterations=self.num_iterations, num_eval_episodes=self.num_eval_episodes, eval_interval=self.eval_interval, log_interval=self.log_interval, train_env=env["train"], eval_env=env["eval"])
                logging.info('Agent training finished')
                #endregion

                #region Agent training results
                greedy = greedy_policy.GreedyPolicy(tf_agent.policy)
                logging.info('Test agent result')
                self.compute_avg_return(self.base_eval_env, greedy, self.num_eval_episodes, display=False)
                #endregion
            except Exception as e:
                logging.error(e)
    
    
    def testReadDumpedData(self):
        logging.info("Opening file")
        with open('result.dump', 'rb') as file:
            data = pickle.load(file)
        logging.info("Everything")
        logging.info(data)
        logging.info("observation")
        logging.info(data.observation)
        logging.info("action")
        logging.info(data.action)
        logging.info("reward")
        logging.info(data.reward)

        logging.info("=======================")
        loops = 10
        logging.info(f"Printing {loops} first items out of {len(data.observation.numpy()[0])}")
        for i in range(loops):
            logging.info(f"========== {i} ==========")
            logging.info("observation")
            logging.info(data.observation.numpy()[0][i])
            logging.info("action")
            logging.info(data.action.numpy()[0][i])
            logging.info("reward")
            logging.info(data.reward.numpy()[0][i])
        logging.info(f"Printing {loops} last items")
        for i in range(loops):
            logging.info(f"========== {i} ==========")
            logging.info("observation")
            logging.info(data.observation.numpy()[0][9000 + i])
            logging.info("action")
            logging.info(data.action.numpy()[0][9000 + i])
            logging.info("reward")
            logging.info(data.reward.numpy()[0][9000 + i])
    
    
    #region common methods for all agents
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb
    # DynamicStepDriver doesn't stop. Try with cuda ?
    def trainAvecDynamicStepDriver(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_steps_per_iteration=10):
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.base_train_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=initial_collect_steps)
        logging.info("Starting to run initial driver collection")
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        initial_collect_driver.run()
        logging.info("Finished to run initial driver collection")
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=64, num_steps=collect_steps_per_iteration).prefetch(3)

        iterator = iter(dataset)
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.base_train_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=collect_steps_per_iteration)
        

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)
        collect_driver.run = common.function(collect_driver.run)
        
        logging.info("Reset the train step")
        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        logging.info("Evaluate the agent's policy once before training")
        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.base_eval_env, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for i in range(num_iterations):
            logging.info(f"Iteration {i+1} out of {num_iterations}")
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_driver.run()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss

            step = tf_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                logging.info('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.base_eval_env, greedy_policy.GreedyPolicy(tf_agent.policy), num_eval_episodes)
                logging.info('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    # Error: One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'.
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def trainAvecJusteReplayBuffer(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_episodes_per_iteration = 2):
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.base_eval_env, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]
        replay_buffer.clear()
        logging.warning(self.num_iterations)
        for i in range(self.num_iterations):
            #logging.info(f"Iteration {i+1} out of {num_iterations}")
            # Collect a few episodes using collect_policy and save to the replay buffer.
            self.collect_episode(
                self.base_train_env, tf_agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

            # Use data from the buffer and update the agent's network.
            experience = replay_buffer.gather_all()
            train_loss = tf_agent.train(experience).loss
            replay_buffer.clear()

            #step = tf_agent.train_step_counter.numpy()

            if i % log_interval == 0:
                logging.info('step = {0}: loss = {1}'.format(i, train_loss))

            if i % eval_interval == 0:
                avg_return = self.compute_avg_return(self.base_eval_env, tf_agent.policy, num_eval_episodes)
                logging.info('step = {0}: Average Return = {1}'.format(i, avg_return))
                returns.append(avg_return)

    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
    def train3(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_steps_per_iteration = 10, train_env = None, eval_env=None):
        if train_env is None:
            train_env = self.base_train_env
        if eval_env is None:
            eval_env = self.base_eval_env
        # Initialize data collection with random tests
        random_policy = random_tf_policy.RandomTFPolicy(
            train_env.time_step_spec(), 
            train_env.action_spec())

        self.collect_data(train_env, random_policy, replay_buffer, steps=2)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=64, 
            #num_steps=collect_steps_per_iteration).prefetch(3) # must be 2 otherwise sends exception
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        returns = []

        for i in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            #for _ in range(collect_steps_per_iteration):
            for _ in range(2):
                self.collect_step(train_env, collect_policy, replay_buffer, i)
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                
                train_loss = tf_agent.train(experience).loss

            #step = tf_agent.train_step_counter.numpy()

            #if i % log_interval == 0:
                #logging.info('step = {0}: loss = {1}'.format(i, train_loss))

            if i % eval_interval == 0:
                logging.info('step = {0}:'.format(i))
                avg_return = self.compute_avg_return(eval_env, greedy_policy.GreedyPolicy(tf_agent.policy), num_eval_episodes)
                returns.append(avg_return)

    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
    def AlwaysGreedy(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_steps_per_iteration = 10):
        policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        
        # Initialize data collection
        self.collect_data(self.base_train_env, policy, replay_buffer, steps=2)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=64, 
            #num_steps=collect_steps_per_iteration).prefetch(3)
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        returns = []

        for i in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            #for _ in range(collect_steps_per_iteration):
            for _ in range(2):
                self.collect_step(self.base_train_env, policy, replay_buffer, i)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                
                train_loss = tf_agent.train(experience).loss

            #step = tf_agent.train_step_counter.numpy()

            #if i % log_interval == 0:
            #    logging.info('step = {0}: loss = {1}'.format(i, train_loss))

            if i % eval_interval == 0:
                logging.info('step = {0}:'.format(i))
                avg_return = self.compute_avg_return(self.base_train_env, policy, num_eval_episodes)
                returns.append(avg_return)

    
    def normal_projection_net(self, action_spec,init_means_output_factor=0.1):
        return normal_projection_network.NormalProjectionNetwork(
            action_spec,
            mean_transform=None,
            state_dependent_std=True,
            init_means_output_factor=init_means_output_factor,
            std_transform=sac_agent.std_clip_transform,
            scale_distribution=True)

    def compute_avg_return(self, environment, policy, num_episodes=5, display=False):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            if display:
                logging.info(f" Episode return : {episode_return}")
            total_return += episode_return
        ret = total_return / num_episodes
        logging.info(f'Average return: {ret}')
        return ret
        # avg_return = total_return / num_episodes
        # return avg_return.numpy()[0]
    
    # From https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def collect_episode(self, environment, policy, replay_buffer, num_episodes):
        episode_counter = 0
        environment.reset()

        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

            if traj.is_boundary():
                episode_counter += 1
    # From https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
    def collect_step(self, environment, policy, buffer, loop_number = None):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
    # From https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
    def collect_data(self, env, policy, buffer, steps):
        for _ in range(steps):
            self.collect_step(env, policy, buffer)
    #endregion


if __name__ == '__main__':
    unittest.main()