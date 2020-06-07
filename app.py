"""
from src.pileFaceEnvironment.pileFace import PileFace
#from pileFaceEnvironment.pileFace import PileFace
import tensorflow as tf
from tf_agents.environments import utils

tf.compat.v1.enable_v2_behavior()

# Validate that the environment works
environment = PileFace()
utils.validate_py_environment(environment, episodes=5)

environment = PileFace()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(10):
    time_step = environment.step(True)
    print(time_step)
    cumulative_reward += time_step.reward

print('Final Reward = ', cumulative_reward)
"""

#import src.agent.agent
import src.agent.MyTFEnvironment