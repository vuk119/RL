from abc import ABCMeta, abstractmethod

import time

import numpy as np

class BaseAlgo(object):
    __metaclass__=ABCMeta

    def __init__(self, env, action_space_shape, state_space_shape):

        self.action_space_shape = action_space_shape if type(action_space_shape) is tuple else (action_space_shape,)
        self.state_space_shape = state_space_shape if type(state_space_shape) is tuple else (state_space_shape,)
        self.state_action_space_shape = self.state_space_shape + self.action_space_shape

        self.env = env

        self.reset()

    def reset(self):
        self.Q = np.zeros(self.state_action_space_shape)

        self.N_state = np.zeros(self.state_space_shape, dtype = int)
        self.N_state_action = np.zeros(self.state_action_space_shape, dtype = int)

    def get_action(self, state, epsilon = None):
        #Epsilon Greedy Strategy

        if epsilon is None:
            epsilon = self.get_epsilon(state)

        if np.random.random() > epsilon:
            return np.unravel_index(np.argmax(self.Q[state]), self.Q[state].shape)
        else:
            return tuple([np.random.choice(action_size) for action_size in self.action_space_shape])

    def sample_episode(self):

        state = self.env.reset()

        #Record the data
        episode = []
        done = False

        #Sample the episode
        while not done:

            action = self.get_action(state=state)
            #If there is a single action do not pass a tuple to the environment
            if len(action)==1:
                action = action[0]

            next_state, reward, done, info = self.env.step(action)

            self.N_state[state] += 1
            self.N_state_action[state][action] += 1
            episode.append([state, action, reward])

            state = next_state

        return episode

    @abstractmethod
    def update(self, episode):
        #Different algorithms require different update strategies
        raise NotImplementedError

    def train(self, n_episodes = 10000, log = True):
        #Usually this function is similar to all algorithms
        #First step is to sample an episode
        #Second step is to make updates based on the sample

        start = time.time()
        for _ in range(n_episodes):
            if log and _%int(0.1*n_episodes)==0:
                print("Completed {} episodes".format(_))
                print("Total time taken {}s".format(time.time()-start))
            episode = self.sample_episode()
            self.update(episode)

        print("Completed the total of {} episodes".format(n_episodes))
        print("The total time taken is {}s".format(time.time()-start))

        return self.Q

    def get_epsilon(self, state):
        #This function should be overriden to adjust specific needs

        base_n = 100
        return base_n/(base_n+self.N_state[state])

    def get_Q(self):
        return self.Q

    def get_V(self):
        axis = tuple(np.arange(-len(self.action_space_shape,0)))
        return np.max(self.Q, axis = axis)

    def get_optimal_policy(self):
        axis = tuple(np.arange(0,len(self.state_space_shape)))
        return np.argmax(self.Q, axis = axis)
