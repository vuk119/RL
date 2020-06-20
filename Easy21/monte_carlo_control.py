"""

Monte Carlo Control Implementation

"""

import time
import pickle
import os

import numpy as np

from base_algo import BaseAlgo


class MonteCarloControl(BaseAlgo):

    def __init__(self, env, action_space_shape, state_space_shape):
        super().__init__(env, action_space_shape, state_space_shape)

        self.reset()

    def sample_episode(self):

        state = self.env.reset()

        #Record the data
        episode = []
        visited_state_action = np.zeros(self.state_action_space_shape, dtype = bool)
        done = False
        cum_reward = 0

        #Sample the episode
        while not done:

            action = self.get_action(state=state)

            #If there is a single action do not pass a tuple to the environment
            if len(action)==1:
                action = action[0]

            next_state, reward, done, info = self.env.step(action)

            if visited_state_action[state][action] == False:
                self.N_state[state] += 1
                self.N_state_action[state][action] += 1
                episode.append([state, action, cum_reward])
                visited_state_action[state][action] = True

            state = next_state
            cum_reward += reward

        #Correct the entries to contain the observed returns
        total_return = cum_reward
        for index, (state, action, cum_reward) in enumerate(episode):
            observed_return = total_return - cum_reward
            episode[index] = (state, action, observed_return)

        return episode

    def update(self, episode):
        for state, action, observed_return in episode:
            self.Q[state][action] = self.Q[state][action] + (observed_return - self.Q[state][action])/self.N_state_action[state][action]

    def play(self):
        policy = self.get_optimal_policy()

        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            self.env.render()
            action = self.get_action(state=state)

            if len(action)==1:
                action=action[0]

            next_state, reward, done, info = self.env.step(action)
            total_reward += reward

        self.env.render()

        print("Finished")
        print("The final reward is {}".format(reward))

    def save_Q(self, name):
        with open(os.path.join('./out',name +'_Q_val.pickle'), 'wb') as f:
            pickle.dump(self.Q, f)

from easy21 import Easy21, Action

env = Easy21()
env_name = 'Easy21'
MC = MonteCarloControl(env, 2, (22,22))


Q = MC.train(n_episodes=1000000)

V = np.max(Q, axis = -1)
relevant_V = V[1:22,1:11]

MC.save_Q(env_name)

import matplotlib.pyplot as plt

policy = np.argmax(Q, axis = -1)
plt.subplot(121)
plt.imshow(V)
plt.subplot(122)
plt.imshow(relevant_V)
plt.show()

from plot_cuts import *

matrix_surf(V[1:22,1:11], xlimits = [1,22], ylimits = [1,11], xlabel = 'Player Sum', ylabel='Dealer Showing', xticks=np.arange(1,22,2), yticks = np.arange(1,11))
