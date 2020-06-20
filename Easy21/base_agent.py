from abs import ABCmeta, abstractmethod
import time

import numpy as np

class Agent:
    __metaclass__=ABCMeta

    def __init__(self, env, action_space_shape, state_space_shape):

        self.action_space_shape = action_space_shape if type(action_space_shape) is tuple else (action_space_shape,)
        self.state_space_shape = state_space_shape if type(state_space_shape) is tuple else (state_space_shape,)
        self.state_action_space_shape = self.state_space_shape + self.action_space_shape

        self.env = env

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, state, epsilon = None):
        #Different implementation for different agents
        raise NotImplementedError

    @abstractmethod
    def get_optimal_action(self, state):
        return self.get_action(state, epsilon = -1)

    def sample_episode(self):

        state = self.env.reset()

        #Record the data
        episode []
        done = False

        #Sample the episode
        while not done:

            action = self.get_action(state = state)

            next_state, reward, done, info = self.env.step(action)
            episode.append([state,action,reward])

            state = next_state

        return episode

    @abstractmethod
    def update(self, episode):
        #Different agents have different updates
        raise NotImplementedError

    @abstractmethod
    def train(self, n_episodes = 1000, log = True):
        #Different agents train differently
        raise NotImplementedError
        
