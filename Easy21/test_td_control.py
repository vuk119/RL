import unittest

import numpy as np

from td_control import *

class TestTDn(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_td_target(self):
        alg = TDn(None, (2,), (2,), n =2)

        states = [0,0,1,1,0]
        actions = [1,1,0,1,0]
        rewards = np.array([1,0,0,1,1])

        '''
        Test SARSA td target
        '''
        alg.algo = 'SARSA'
        alg.Q = np.array([[1,2], [3,4]])

        td_target = 1+alg.gamma*0+alg.gamma**2 * alg.Q[1,0]
        self.assertAlmostEqual(alg.get_td_target(0,rewards,states,actions),td_target)

        td_target = 0+alg.gamma*0+alg.gamma**2 * alg.Q[1,1]
        self.assertAlmostEqual(alg.get_td_target(1,rewards,states,actions),td_target)

        td_target = 0+alg.gamma*1+alg.gamma**2 * alg.Q[0,0]
        self.assertAlmostEqual(alg.get_td_target(2,rewards,states,actions),td_target)


        '''
        Test QLearning td target
        '''
        alg.algo = 'QLearning'
        alg.Q = np.array([[1,2], [3,4]])

        td_target = 1+alg.gamma*0+alg.gamma**2 * alg.Q[1,1]
        self.assertAlmostEqual(alg.get_td_target(0,rewards,states,actions),td_target)

        td_target = 0+alg.gamma*0+alg.gamma**2 * alg.Q[1,1]
        self.assertAlmostEqual(alg.get_td_target(1,rewards,states,actions),td_target)

        td_target = 0+alg.gamma*1+alg.gamma**2 * alg.Q[0,1]
        self.assertAlmostEqual(alg.get_td_target(2,rewards,states,actions),td_target)


class TDLambda(unittest.TestCase):

    def setUp(self):
        algo = TDLambda()

    def test_get_td_target(self):
        pass
