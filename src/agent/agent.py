from src.pileFaceEnvironment.pileFace import PileFace
from src.environment.interface import interface
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.agents import *
import random
tf.compat.v1.enable_v2_behavior()

def doNRandomSteps(numberOfSteps, environment, verbose: bool = True):
    # Validate that the environment works
    # utils.validate_py_environment(environment, episodes=5)
    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    for _ in range(numberOfSteps):
        time_step = environment.step(random.randint(0, 200) / 100)
        if verbose:
            print(time_step)
        cumulative_reward += time_step.reward

    print('Final Reward = ', cumulative_reward)

def doNSteps(numberOfSteps: int, environment, verbose: bool = True):
    # Validate that the environment works
    # utils.validate_py_environment(environment, episodes=5)
    time_step = environment.reset()
    print(time_step)
    cumulative_reward: int64 = time_step.reward
    rewards = [time_step.reward]
    action = 1

    for i in range(numberOfSteps):
        evolution = 0.9 if random.random() < 0 else 1.1
        time_step = environment.step(action*evolution)
        if verbose:
            print("Action: ", str(action))
            print(time_step)
            print("Reward: ", time_step.reward, "\n")
        cumulative_reward += time_step.reward
        if i>0 and (time_step.reward > (cumulative_reward/i)):
            action *= evolution
        elif i>0:
            action /= evolution
    print('Final Action = ', action)
    print('Final Reward = ', cumulative_reward)
    print('Average Reward = ', cumulative_reward/numberOfSteps)

steps = 10000
environment = interface(maxSteps=steps)
doNSteps(numberOfSteps=steps, environment=environment, verbose=False)