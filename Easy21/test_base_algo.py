import unittest

from base_algo import BaseAlgo

class TestEasy21(unittest.TestCase):

    def setUp(self):
        self.algo1 = BaseAlgo(None, 5, (2,3))
        self.algo2 = BaseAlgo(None, (5,2), (3,))
        self.algo3 = BaseAlgo(None, (5,2), (3,2,3))

    def test_get_action(self):

        state = tuple([0]*len(self.algo1.state_space_shape))
        self.assertEqual(len(self.algo1.get_action(state)), len(self.algo1.action_space_shape))


        state = tuple([0]*len(self.algo2.state_space_shape))
        self.assertEqual(len(self.algo2.get_action(state)), len(self.algo2.action_space_shape))


        state = tuple([0]*len(self.algo3.state_space_shape))
        self.assertEqual(len(self.algo3.get_action(state)), len(self.algo3.action_space_shape))

    def test_get_V(self):
        pass 

if __name__ == '__main__':
    unittest.main()
