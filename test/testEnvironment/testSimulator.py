import unittest
from tf_agents.environments import utils
"""
from src.environment.simulator import simulator

class testSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = simulator()
    def testGetPlaces(self):
        places = self.sim.allPlaces()
        self.assertEqual(len(places), 10)
    
    def testGetPlacesDemand(self):
        places = self.sim.allPlaces()
        for i in range(len(places)):
            self.sim.getDemand(i)
        # If no exception raised, we're fine
        self.assertTrue(True)
    def testGetPlacesDemand2(self):
        places = self.sim.allPlaces()
        with self.assertRaises(IndexError):
            for i in range(len(places)+1):
                self.sim.getDemand(i)

if __name__ == '__main__':
    unittest.main()
"""
from src.environment.PyEnv import PyEnv

class test(unittest.TestCase):
    def setUp(self):
        self.size = 10
        self.duration = 10
        self.env = PyEnv(self.size, self.duration)

    def testValidate(self):
        utils.validate_py_environment(self.env, episodes=5)

    """
    def testPlay(self):
        time_step = self.env.reset()
        #step = np.array(.5, dtype=np.float32)

        print(time_step)
        cumulative_reward = time_step.reward

        for _ in range(3):
            time_step = self.env.step(.8)
            print(time_step)
            cumulative_reward += time_step.reward

        cumulative_reward += time_step.reward
        print('Final Reward = ', cumulative_reward)
    """

    def testUselessAction(self):
        action = []
        for i in range(self.size):
            action.append(i)
        keep = True
        stateEstimation = 0
        while keep:
            result = self.env._step(action)
            if False:
                print("Observation: ", result[0])
                print("Reward: ", result[1])
            keep = result[2]
        print("Finished")        