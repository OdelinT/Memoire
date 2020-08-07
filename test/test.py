#region imports
import unittest
from tf_agents.environments import utils
import tensorflow as tf
from tf_agents.networks import q_network, normal_projection_network, actor_distribution_network
from tf_agents.agents.dqn import dqn_agent
from environment.PyEnv import PyEnv
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

import numpy as np
#endregion

# Test cases are commented and decommented according to what I was studying during commit.

class test(unittest.TestCase):
    def setUp(self):
        coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')
        self.env = PyEnv()
        self.env2 = PyEnv()
        self.env3 = PyEnv()
        self.tf_env = tf_py_environment.TFPyEnvironment(self.env)
        self.train_env = tf_py_environment.TFPyEnvironment(self.env2)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.env3)

    """
    def testMyValidate(self):
        self.env._reset()
        #region Test environment parameters generation
        logging.info("Test environment parameters generation")
        for value in self.env.placesSizes:
            if value < 0:
                logging.error("At least one place size < 0")
        for value in self.env.productsCosts:
            if value < 0:
                logging.error("At least one product cost < 0")
        for value in self.env.productsUsualMarginRates:
            if value < 0:
                logging.error("At least one margin rate < 0")
            if value > 1:
                logging.error("At least one margin rate > 1")
        for value in self.env.productsUsualBuyingRates:
            if value <= 0:
                logging.error("At least one buying rate <= 0")
        for i in range(len(self.env.productsUsualPrices)):
            if self.env.productsUsualPrices[i] < self.env.productsCosts[i]:
                logging.error("At least one usual price smaller than product cost")
        #endregion

        #region Test environment credibility with different prices
        logging.info("Test environment credibility with different prices")
        observation = self.env._step(self.env.productsCosts)
        logging.info(f"Product costs: {observation}")
        costReward = observation.reward
        if observation.reward != 0:
            logging.error("Error: If we sell at product cost, there should be no margin")

        observation = self.env._step(self.env.productsCosts * 2)
        logging.info(f"Product costs *2: {observation}")
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            logging.error("Error: If we sell at more than product cost, there should be a margin")

        observation = self.env._step(self.env.productsCosts * 10)
        logging.info(f"Product costs *10: {observation}")
        if observation.reward <= 0 and np.sum(observation.observation) > 0:
            logging.error("Error: If we sell at more than product cost, there should be a margin")

        observation = self.env._step(self.env.productsUsualPrices)
        logging.info(f"Usual buying price: {observation}")
        usualReward = observation.reward

        observation = self.env._step(self.env.productsUsualPrices * 2)
        logging.info(f"Usual buying price *2: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.env._step(self.env.productsUsualPrices * 4)
        logging.info(f"Usual buying price *4: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.env._step(self.env.productsUsualPrices * 10)
        logging.info(f"Usual buying price *10: {observation}")
        logging.info(f"Compared to usual: {observation.reward / usualReward}")

        observation = self.env._step(np.zeros(100))
        logging.info(f"0: {observation}")
        if observation.reward >= 0.:
            logging.error("Error: If prices are 0, we should sell and have a deficit")

        observation = self.env._step(np.zeros(100) + 1)
        logging.info(f"1: {observation}")
        if observation.reward >= 0:
            logging.error("Error: If prices are 1 and average cost 10, we should sell and have a deficit")

        observation = self.env._step(np.zeros(100) + 1000)
        logging.info(f"1000: {observation}")
        if observation.reward > usualReward:
            logging.error("Error: An unusual price such as 1000 shouldn't cause a best result than the usual price. Error ratio: ", observation.reward / usualReward)
        #endregion
    
    def testValidate(self):
        utils.validate_py_environment(self.env, episodes=5)
    """
    
    
    # No error but reward=0 almost every time
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def testReinforceActorDistributionNet(self):
        logging.info("Starting testReinforceActorDistributionNet")
        #region Hyperparameters from the example of the documentation
        num_iterations = 1000 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 50 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 100 # @param {type:"integer"}
        #endregion

        #region Agent initialization
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
        #endregion

        collect_policy = tf_agent.collect_policy

        #logging.info(self.compute_avg_return(self.train_env, collect_policy))
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_capacity)

        #region Agent training
        
        #regionAdditional parameters
        initial_collect_steps = 10
        #endregion
        
        #No complete episode found. REINFORCE requires full episodes to compute losses
        #self.trainAvecDynamicStepDriver(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)
        
        # Return = 0
        #self.trainAvecJusteReplayBuffer(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)

        # Return = 0
        self.train3(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)
        
        #endregion
    
    """
    # Error: One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'.
    def testTd3(self):
        logging.info("Starting testTd3")
        #region Hyperparameters from the example of the documentation
        num_iterations = 250 # @param {type:"integer"}
        collect_episodes_per_iteration = 2 # @param {type:"integer"}
        replay_buffer_capacity = 2000 # @param {type:"integer"}

        fc_layer_params = (100,)

        learning_rate = 1e-3 # @param {type:"number"}
        log_interval = 25 # @param {type:"integer"}
        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 50 # @param {type:"integer"}
        #endregion
        
        #region Agent initialization
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
        
        tf_agent = tf_agents.agents.Td3Agent(
            time_step_spec = self.tf_env.time_step_spec(), 
            action_spec = self.tf_env.action_spec(),
            actor_network = actor_net,
            critic_network = critic_net,
            actor_optimizer = actor_optimizer,
            critic_optimizer = critic_optimizer)
        tf_agent.initialize()
        #endregion

        collect_policy = tf_agent.collect_policy

        logging.info(self.compute_avg_return(self.train_env, collect_policy))
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_capacity)

        #region Agent training
        
        #regionAdditional parameters
        initial_collect_steps = 0
        #endregion
        
        #DynamicStepDriver takes too long time
        #self.trainAvecDynamicStepDriver(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, 0)

        # One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'
        #self.trainAvecJusteReplayBuffer(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)

        # call() missing 1 required positional argument: 'network_state'
        self.train3(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)
        #endregion
    """
        
    """
    # Works at least until initialization
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb
    def testSAC(self):
        logging.info("Starting testSAC")

        #region Hyperparameters from the example of the documentation
        # use "num_iterations = 1e6" for better results,
        # 1e5 is just so this doesn't take too long. 
        num_iterations = 500 # @param {type:"integer"}

        initial_collect_steps = 100 # @param {type:"integer"} 
        collect_steps_per_iteration = 1 # @param {type:"integer"}
        replay_buffer_capacity = 100 # @param {type:"integer"}

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

        log_interval = 50 # @param {type:"integer"}

        num_eval_episodes = 10 # @param {type:"integer"}
        eval_interval = 100 # @param {type:"integer"}

        # parameters from other documentation examples
        collect_episodes_per_iteration = 2
        
        #endregion
        
        #region Agent initialization
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
        #endregion

        #region Set training variables
        logging.info("Set training variables")
        eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        collect_policy = tf_agent.collect_policy

        logging.info(self.compute_avg_return(self.train_env, eval_policy))
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=replay_buffer_capacity)
        #endregion
        
        #region Agent training
        
        #DynamicStepDriver takes too long time
        #self.trainAvecDynamicStepDriver(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)
        
        # One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'
        #self.trainAvecJusteReplayBuffer(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)

        #Works! but very bad results and doesn't appear to be improving
        #self.train3(tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval)
        
        #endregion
    """

    #region common methods for all agents
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/7_SAC_minitaur_tutorial.ipynb
    # DynamicStepDriver doesn't stop. Try with cuda ?
    def trainAvecDynamicStepDriver(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_steps_per_iteration=10):
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=initial_collect_steps)
        logging.info("Starting to run initial driver collection")
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        initial_collect_driver.run()
        logging.info("Finished to run initial driver collection")
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)

        iterator = iter(dataset)
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
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
        avg_return = self.compute_avg_return(self.eval_env, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for i in range(num_iterations):
            logging.info(f"Iteration {i+1} out of {num_iterations}")
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_driver.run()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience)

            step = tf_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, eval_policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    # Error: One of the Tensors in `experience` has a time axis dim value '33', but we require dim value '2'.
    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/6_reinforce_tutorial.ipynb
    def trainAvecJusteReplayBuffer(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_episodes_per_iteration = 10):
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.eval_env, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for i in range(num_iterations):
            logging.info(f"Iteration {i+1} out of {num_iterations}")
            # Collect a few episodes using collect_policy and save to the replay buffer.
            self.collect_episode(
                self.train_env, tf_agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

            # Use data from the buffer and update the agent's network.
            experience = replay_buffer.gather_all()
            train_loss = tf_agent.train(experience)
            replay_buffer.clear()

            step = tf_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                logging.info('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, tf_agent.policy, num_eval_episodes)
                logging.info('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    # https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
    def train3(self, tf_agent, collect_policy, replay_buffer, initial_collect_steps, num_iterations, num_eval_episodes, eval_interval, log_interval, collect_steps_per_iteration = 10):
        random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                        self.train_env.action_spec())

        self.collect_data(self.train_env, random_policy, replay_buffer, steps=100)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=64, 
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(self.eval_env, tf_agent.policy, num_eval_episodes)
        returns = [avg_return]

        for _ in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                self.collect_step(self.train_env, tf_agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss

            step = tf_agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(self.eval_env, tf_agent.policy, num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

    
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
        logging.info("Starting to compute")
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            logging.info(f" Episode return : {episode_return}")
            total_return += episode_return
        return total_return / num_episodes
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
    def collect_step(self, environment, policy, buffer):
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


