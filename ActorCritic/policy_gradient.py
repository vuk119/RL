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

from model import Model
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


class PolicyGradient(Model):

    def __init__(self, observation_space_size, action_space_size, name=None,
                 env_name=None, model_config=None, play_mode=False):

        if name is None:
            name = "Unnamed-PolicyGradient"
        super(PolicyGradient, self).__init__(observation_space_size, action_space_size,
                                             name, env_name, model_config, play_mode)

    def build_model(self):

        self.policy_net = Actor(self.observation_space_size, self.action_space_size)

        if self.model_config is None:
            self.gamma = 0.99
            self.optimizer = optim.Adam(self.policy_net.parameters())
            self.loss = nn.MSELoss()
            self.get_epsilon = self.get_epsilon_default
        else:
            pass

    def save_checkpoint(self, n=0, filepath=None):
        """
        n - number of epoch / episode or whatever is used for enumeration
        """

        # TO DO: ADD OTHER RELEVANT PARAMETERS
        checkpoint = {'policy': self.policy_net.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        super(PolicyGradient, self).save_checkpoint(n, filepath, checkpoint)

    def load_checkpoint(self, filepath):
        # TO DO: ADD OTHER RELEVANT parameters
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def actor_update(self, Q, action, mu):
        self.actor_optimizer.zero_grad()
        actor_loss = self.actor_loss(action, mu)
        gradient_term = Q * actor_loss
        gradient_term.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def update(self, sample, prepare_state=None):
        """
        prepare_state is a function that does feature engineering on the plain state
        """

        sample = self.monte_carlo_returns(sample)
        episode_running_loss = []

        for state, action, Q in sample:
            if prepare_state is not None:
                state = prepare_state(state)

            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)

            mu = self.policy_net(state)

            loss = self.actor_update(Q, action, mu)
            episode_running_loss.append(loss)

        return episode_running_loss


if True:
    from continuous_cart_pole_env import ContinuousCartPoleEnv

    env_name = 'ContinuousCartPoleEnv'
    env = ContinuousCartPoleEnv()

    print(env.action_space.shape)
    print(env.observation_space.shape)
    print(env.action_space.low, env.action_space.high)
    model = PolicyGradient(4, 1)

    n_episode = 5
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
        model.update(history)

    print("DONE")
