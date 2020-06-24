import gym

import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F

if False:
    torch
    nn
    F

env = gym.make("MountainCarContinuous-v0")


class Critic(nn.Module):

    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nh1 = 40
        self.nh2 = 40
        self.fc1 = nn.Linear(input_size, self.nh1)
        self.fc2 = nn.Linear(self.nh1, self.nh2)
        self.fc3 = nn.Linear(self.nh2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nh1 = 400
        self.nh2 = 400
        self.fc1 = nn.Linear(input_size, self.nh1)
        self.fc2 = nn.Linear(self.nh1, self.nh2)
        self.mu = nn.Linear(self.nh2, output_size)
        self.sigma = nn.Linear(self.nh2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_dist(self, state):
        state = self.forward(state)
        mu = self.mu(state)
        sigma = self.sigma(state)
        sigma = F.softplus(sigma) + 1e-5
        return mu.item(), sigma.item()

    def get_action(self, state):
        # Only single action envs at the moment!
        state = self.forward(state)
        mu = self.mu(state)
        sigma = self.sigma(state)
        sigma = F.softplus(sigma) + 1e-5
        actions = torch.normal(mu, sigma)
        actions = torch.clamp(actions, env.action_space.low[0], env.action_space.high[0])
        return actions


def test_actor():
    cri = Critic(5, 1)
    inpt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    assert type(cri.get_action(inpt).item()) is float


def test_critic():
    act = Actor(5, 1)
    inpt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    assert type(act.forward(inpt).item()) is float


def define_optimizers(actor, critic):
    lr_actor = 0.00001
    lr_critic = 0.00056

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    return optimizer_actor, optimizer_critic


actor = Actor(env.observation_space.shape[0], 1)
critic = Critic(env.observation_space.shape[0], 1)
lr_actor = 0.00001
lr_critic = 0.00056

optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)


gamma = 0.99
num_episodes = 1

episode_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action = actor.get_action(state)

        next_state, reward, done, info = env.step([action.item()])

        steps += 1
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32)

        V_state = critic(state)
        V_next_state = critic(next_state)
        target = torch.tensor(reward, dtype=torch.float32) + torch.mul(gamma, critic(next_state))
        td_error = target - V_state

        critic_loss = nn.MSELoss(target, V_state)
        actor_loss =
