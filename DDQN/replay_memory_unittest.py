import unittest

import torch

from replay_memory import ReplayMemory


class TestReplayMemory(unittest.TestCase):

    def test_sample(self):
        pass

    def test_push(self):
        rpl = ReplayMemory(3, 2)
        rpl.push(([1, 1], 1, 2, [2, 2]))
        rpl.push(([1, 1], 1, 2, [2, 2]))
        rpl.push(([1, 1], 1, 2, [2, 2]))
        rpl.push(([5, 5], 6, 3, [3, 3]))

        state_memory = torch.tensor([[5., 5.],
                                     [1., 1.],
                                     [1., 1.]])

        action_memory = torch.tensor([[6],
                                      [1],
                                      [1]])

        reward_memory = torch.tensor([[3.],
                                      [2.],
                                      [2.]])

        next_state_memory = torch.tensor([[3., 3.],
                                          [2., 2.],
                                          [2., 2.]])

        self.assertTrue(torch.all(torch.eq(rpl.state_memory, state_memory)))
        self.assertTrue(torch.all(torch.eq(rpl.action_memory, action_memory)))
        self.assertTrue(torch.all(torch.eq(rpl.reward_memory, reward_memory)))
        self.assertTrue(torch.all(torch.eq(rpl.next_state_memory, next_state_memory)))
