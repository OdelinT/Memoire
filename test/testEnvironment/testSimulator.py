import unittest
from tf_agents.environments import utils
from src.environment.PyEnv import PyEnv

class test(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.duration = 10
        # Overriding the default value of the parameters breaks utils.validate_py_environment
        # because it will use the default parameters when sending actions to the environment, regardless the actual parameters
        # self.env = PyEnv(self.size, self.duration)
        self.env = PyEnv()

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