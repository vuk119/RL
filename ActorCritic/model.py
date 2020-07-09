import random
import math
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


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
    time


class Model(ABC):
    """
    Base model for all policy gradient based models.

    play_mode is True if you are loading the model.
    """

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

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    def get_epsilon_default(self):
        return 0.01

    def get_action(self, state, sigma, prepare_state=None):
        """
        -prepare_state is a function that does feature engineering on the plain state

        -this is the default for PG, A2C ... Override if needed
        """
        if prepare_state is not None:
            state = prepare_state(state)

        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mu = self.policy_net(state).item()
        if sigma != 0:
            action = np.random.normal(mu, sigma, 1)[0]
        else:
            action = mu
        action = max(action, -1)
        action = min(action, 1)

        return action

    def save_checkpoint(self, n=0, filepath=None, checkpoint=None):
        """
        n - number of epoch / episode or whatever is used for enumeration
        """

        # TO DO: ADD OTHER RELEVANT PARAMETERS
        if checkpoint is None:
            checkpoint = {'policy': self.policy_net.state_dict(),
                          'optimizer': self.optimizer.state_dict()}
        if filepath is None:
            path = os.path.join(self.dir, 'checkpoints')
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.join(path, 'n' + str(n))
        torch.save(checkpoint, filepath)

    @abstractmethod
    def load_checkpoint(self, filepath):
        raise NotImplementedError

    def monte_carlo_returns(self, history):
        """
        input: [[state, action, reward] over steps]
        output: [[state, action, Q[state, action]] over steps]
        """

        for i in range(len(history) - 2, -1, -1):
            history[i][2] += history[i + 1][2]

        return history

    @abstractmethod
    def update(self, sample, prepare_state=None):
        raise NotImplementedError
