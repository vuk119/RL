import random

import torch


class ReplayMemory(object):

    def __init__(self, capacity, observation_space_size=4):
        self.capacity = capacity
        self.position = 0
        self.length = 0

        self.state_memory = torch.zeros(capacity, observation_space_size)
        self.action_memory = torch.zeros(capacity, 1, dtype=torch.long)
        self.reward_memory = torch.zeros(capacity, 1)
        self.next_state_memory = torch.zeros(capacity, observation_space_size)
        self.terminal_memory = torch.zeros(capacity, 1, dtype=torch.long)

    def push(self, episode):
        state, action, reward, next_state, terminal = episode
        self.state_memory[self.position, :] = torch.tensor(state, dtype=torch.float32)
        self.action_memory[self.position, :] = torch.tensor(action, dtype=torch.long)
        self.reward_memory[self.position, :] = torch.tensor(reward, dtype=torch.float32)
        self.next_state_memory[self.position, :] = torch.tensor(next_state, dtype=torch.float32)
        self.terminal_memory[self.position, :] = torch.tensor(terminal, dtype=torch.bool)

        self.position = (self.position + 1) % self.capacity
        self.length += 1

        assert self.position == self.length % self.capacity

    def sample(self, batch_size):
        sample = random.sample(range(self.__len__()), batch_size)
        return self.state_memory[sample, :], self.action_memory[sample, :], self.reward_memory[sample, :], self.next_state_memory[sample, :], self.terminal_memory[sample, :]

    def __len__(self):
        return self.length if self.length < self.capacity else self.capacity
