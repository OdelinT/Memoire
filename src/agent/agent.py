from src.pileFaceEnvironment.pileFace import PileFace
from src.environment.interface import interface
import tensorflow as tf
from tf_agents.environments import utils

tf.compat.v1.enable_v2_behavior()

def doNSteps(numberOfSteps, environment):
    # Validate that the environment works
    utils.validate_py_environment(environment, episodes=5)
    time_step = environment.reset()
    print(time_step)
    cumulative_reward = time_step.reward

    for _ in range(numberOfSteps):
        time_step = environment.step(True)
        print(time_step)
        cumulative_reward += time_step.reward

    print('Final Reward = ', cumulative_reward)

environment = interface()
doNSteps(numberOfSteps=100, environment=environment)