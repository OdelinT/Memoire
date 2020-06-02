import unittest
# import environment
from environment import simulator
from environment.simulator import simulator

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