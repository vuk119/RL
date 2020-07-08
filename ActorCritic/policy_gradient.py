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

from actor import Actor

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
    Actor


class PolicyGradient:

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

        self.policy_net = Actor(self.observation_space_size, self.action_space_size)

        if self.model_config is None:
            self.gamma = 0.99
            self.optimizer = optim.Adam(self.policy_net.parameters())
            self.loss = nn.MSELoss()
            self.get_epsilon = self.get_epsilon_default
        else:
            pass

    def get_epsilon_default(self):
        return 0.01

    def get_action(self, state, sigma, prepare_state=None):
        """
        prepare_state is a function that does feature engineering on the plain state
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

    def save_checkpoint(self, n=0, filepath=None):
        """
        n - number of epoch / episode or whatever is used for enumeration
        """

        # TO DO: ADD OTHER RELEVANT PARAMETERS
        checkpoint = {'policy': self.policy_net.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
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
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def process_episode_history(self, history):
        """
        input: [[state, action, reward] over steps]
        output: [[state, action, Q[state, action]] over steps]
        """

        for i in range(len(history) - 2, -1, -1):
            history[i][2] += history[i + 1][2]

        return history

    def update(self, history, prepare_state=None):
        """
        prepare_state is a function that does feature engineering on the plain state
        """

        history = self.process_episode_history(history)
        episode_running_loss = []

        for state, action, Q in history:
            if prepare_state is not None:
                state = prepare_state(state)

            mu = self.policy_net(torch.tensor(state, dtype=torch.float32))

            self.optimizer.zero_grad()
            loss = Q * self.loss(torch.tensor(action, dtype=torch.float32), mu)
            loss.backward()
            self.optimizer.step()
            episode_running_loss.append(loss.item())

        return episode_running_loss


if True:
    from continuous_cart_pole_env import ContinuousCartPoleEnv

    env_name = 'ContinuousCartPoleEnv'
    env = ContinuousCartPoleEnv()

    print(env.action_space.shape)
    print(env.observation_space.shape)
    print(env.action_space.low, env.action_space.high)
    model = PolicyGradient(4, 1)

    n_episode = 1000
    sigma_max = 1
    sigma_min = 0.05
    rewards = []
    start = time.time()

    for i_episode in range(n_episode):

        if i_episode % 100 == 99:
            print("\n")
            print("Completed:", i_episode + 1, "episodes")
            print("The time taken is:", time.time() - start, "s")
            print("The average reward so far:", np.mean(rewards))
            print("The average reward for the last 100 episodes:", np.mean(rewards[-100:-1]))

        sigma = max(sigma_max * (1 - 4 * i_episode / n_episode), sigma_min)

        state = env.reset()
        done = False
        history = []
        total_reward = 0

        # Play the episode
        while done is not True:
            action = model.get_action(state, sigma)

            next_state, reward, done, info = env.step(np.array([action]))

            history.append([state, action, reward])

            total_reward += reward
            state = next_state

            # Cap the maximum reward, otherwise it'll take forever after it learns the policy
            if total_reward > 300:
                break
        rewards.append(total_reward)
        model.update()
