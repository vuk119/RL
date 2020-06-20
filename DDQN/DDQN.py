import random
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_memory import ReplayMemory
from torch_net import TorchNet

if False:
    random
    math
    os
    np
    plt
    nn
    optim
    F
    SummaryWriter
    torch


class DDQN:

    def __init__(self, observation_space_size, action_space_size, name=None, env_name=None, model_config=None, play_mode=False):

        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        self.model_config = model_config

        self.build_model()
        self.writer = SummaryWriter()

        if play_mode is False:
            if name is None:
                self.name = "Unnamed"
            else:
                self.name = name
            if env_name is None:
                self.env_name = "Unnamed Env"
            else:
                self.env_name = env_name
            self.dir = self.name + '-' + self.env_name + '-' + str(round(time.time()))
            self.dir = os.path.join('./out', self.dir)
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)

    def build_model(self):

        self.policy_net = TorchNet(self.observation_space_size, self.action_space_size)
        self.target_net = TorchNet(self.observation_space_size, self.action_space_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.model_config is None:
            self.batch_size = 32
            self.gamma = 0.99
            self.memory = ReplayMemory(1000, self.observation_space_size)
            self.optimizer = optim.Adam(self.policy_net.parameters())
            self.loss = nn.MSELoss()
            self.get_epsilon = self.get_epsilon_default
        else:
            pass

    def get_epsilon_default(self):
        return 0.01

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon()
        state = torch.tensor(state, dtype=torch.float32)
        sample = random.random()

        if sample > epsilon:
            with torch.no_grad():
                Q = self.policy_net(state)
                _, action = torch.max(Q, -1)
                return action.item()
        else:
            return random.choice(range(self.action_space_size))

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_from_batch(self):

        if self.batch_size > len(self.memory):
            return

        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample(
            self.batch_size)

        predicted_Q = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        next_Q = self.target_net(next_state_batch).max(1)[0].detach()
        next_Q = torch.mul(next_Q, (1 - terminal_batch).squeeze(1))
        target_Q = torch.mul(self.gamma, next_Q) + reward_batch.squeeze(1)

        loss = self.loss(predicted_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_memory(self, episode):
        self.memory.push(episode)

    def save_checkpoint(self, n=0, filepath=None):
        """
        n - number of epoch / episode or whatever is used for enumeration
        """

        # TO DO: ADD OTHER RELEVANT PARAMETERS
        checkpoint = {'policy_net': self.policy_net.state_dict(),
                      'target_net': self.target_net.state_dict(), 'optimizer': self.optimizer.state_dict()}
        if filepath is None:
            path = os.path.join(self.dir, 'checkpoints')
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, 'n' + str(n))
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        # TO DO: ADD OTHER RELEVANT parameters
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
