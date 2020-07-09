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
from critic import Critic


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
    Critic


class ActorCritic(Model):

    def __init__(self, observation_space_size, action_space_size, name=None, env_name=None, model_config=None, play_mode=False):

        if name is None:
            name = "Unnamed-ActorCritic"
        super(ActorCritic, self).__init__(observation_space_size, action_space_size,
                                          name, env_name, model_config, play_mode)

    def build_model(self):

        self.policy_net = Actor(self.observation_space_size, self.action_space_size)
        self.critic_net = Critic(self.observation_space_size)

        if self.model_config is None:
            self.gamma = 0.99

            self.actor_optimizer = optim.Adam(self.policy_net.parameters())
            self.actor_loss = nn.MSELoss()

            self.critic_optimizer = optim.Adam(self.critic_net.parameters())
            self.critic_loss = nn.MSELoss()

            self.get_epsilon = self.get_epsilon_default
        else:
            pass

    def save_checkpoint(self, n=0, filepath=None):
        """
        n - number of epoch / episode or whatever is used for enumeration
        """

        # TO DO: ADD OTHER RELEVANT PARAMETERS
        checkpoint = {'policy': self.policy_net.state_dict(),
                      'critic': self.critic_net.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        super(ActorCritic, self).save_checkpoint(n, filepath, checkpoint)

    def load_checkpoint(self, filepath):
        # TO DO: ADD OTHER RELEVANT parameters
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy'])
        self.critic_net.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def prepare_sample(self, sample):
        sample = np.array(sample)
        states = torch.tensor(sample[:, 0], dtype=torch.float32)
        actions = torch.tensor(sample[:, 1], dtype=torch.float32)
        rewards = torch.tensor(sample[:, 2], dtype=torch.float32)
        next_states = torch.tensor(sample[:, 3], dtype=torch.float32)
        dones = torch.tensor(sample[:, 4], dtype=torch.int32)

        return states, actions, rewards, next_states, dones

    def critic_update(self, V, V_target):
        self.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(V, V_target)
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def actor_update(self, advantages, actions, mus):
        self.actor_optimizer.zero_grad()
        actor_loss = advantages * self.actor_loss(actions, mus)
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def update(self, sample, prepare_state=None):
        actor_running_loss = []
        critic_running_loss = []

        for state, action, reward, next_state, done in sample:
            if prepare_state is not None:
                state = prepare_state(state)
                next_state = prepare_state(next_state)

            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.float32)

            # Update Critic
            V = self.critic_net.forward(state)
            V_target = torch.tensor([reward], dtype=torch.float32)
            if done is False:
                V_target += self.gamma * self.critic_net.forward(next_state)

            critic_loss = self.critic_update(V, V_target)
            critic_running_loss.append(critic_loss)

            # Update Actor
            advantage = (V_target - V).detach()
            mu = self.policy_net(state)

            actor_loss = self.actor_update(advantage, action, mu)
            actor_running_loss.append(actor_loss)

        return actor_running_loss, critic_running_loss

    def batch_update(self, sample, prepare_state=None):
        actor_running_loss = []
        critic_running_loss = []

        states, actions, rewards, next_states, dones = self.prepare_sample(sample)

        # Update Critic
        V = self.critic_net.forward(states)
        V_target = rewards + self.gamma * self.critic_net.forward(next_states) * (1 - dones)

        critic_loss = self.critic_update(V, V_target)
        critic_running_loss.append(critic_loss)

        # Update Actor
        advantage = (V_target - V).detach()
        mu = self.policy_net(states)

        actor_loss = self.actor_update(advantage, actions, mu)
        actor_running_loss.append(actor_loss)

        return actor_running_loss, critic_running_loss
