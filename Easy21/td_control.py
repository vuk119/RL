"""

TD(n) and TD(Lambda) Control implementation

In particular SARSA + QLearning implementation

"""

import time
import pickle
import os

import numpy as np

from base_algo import BaseAlgo

class TDn(BaseAlgo):

    def __init__(self, env, action_space_shape, state_space_shape, gamma = 0.9, alpha = 0.01, n = 0, algo = 'SARSA'):
        '''
        algo should be set to either 'SARSA' or 'QLearning'
        '''
        super().__init__(env, action_space_shape, state_space_shape)

        self.alpha = alpha
        self.n = n
        self.gamma = gamma

        if algo!='SARSA' and algo!='QLearning':
            assert False, "The algo variable must be set to either SARSA or QLearning"
        self.algo = algo

        self.mask = np.array([1] + [self.gamma**i for i in range(1,self.n)])

    def get_td_target(self, index, rewards, states, actions):
        td_target = np.dot(self.mask, rewards[index: index+self.n])
        if self.algo == 'SARSA':
            td_target+= self.gamma**self.n*self.Q[states[index+self.n]][actions[index+self.n]]
        if self.algo == 'QLearning':
            optimal_action = self.get_action(states[index+self.n],epsilon=0)
            td_target+= self.gamma**self.n*self.Q[states[index+self.n]][optimal_action]

        return td_target

    def update(self, episode):
        #The episode is of the form (S_t, A_t, R_{t+1})

        states = []
        actions = []
        rewards = []

        for state, action, reward in episode:
            states.append(state)
            actions.append(action)
            rewards.append(reward)

        rewards = np.array(rewards)

        for index, (state,action,reward) in enumerate(episode):

            #Can't calculate td target for these
            if index + self.n >= len(episode):
                break

            td_target = self.get_td_target(index, rewards, states, actions)
            self.Q[state][action] += self.alpha*(td_target - self.Q[state][action])

class TDLambda(BaseAlgo):

    def __init__(self, env, action_space_shape, state_space_shape, gamma = 0.9, alpha = 0.01, lmbd = 0.7, algo = 'SARSA'):
        super().__init__(env, action_space_shape, state_space_shape)

        self.gamma = gamma
        self.alpha = 0.01
        self.lmbd = lmbd

        if algo!='SARSA' and algo!='QLearning':
            assert False, "The algo variable must be set to either SARSA or QLearning"
        self.algo = algo

    def train(self, n_episodes = 10000, log = True):
        #Different than for the other algorithms
        norms = []

        start = time.time()
        for _ in range(n_episodes):
            if log and _%int(0.1*n_episodes)==0:
                print("Completed {} episodes".format(_))
                print("Total time taken {}s".format(time.time()-start))

            E = np.zeros(self.state_action_space_shape)


            #Initial state and action
            state = self.env.reset()
            action = self.get_action(state=state)
            #If there is a single action do not pass a tuple to the environment
            if len(action)==1:
                action = action[0]

            done = False

            while not done:
                next_state, reward, done, info = self.env.step(action)

                next_action = self.get_action(state=next_state)
                #If there is a single action do not pass a tuple to the environment
                if len(next_action)==1:
                    next_action = next_action[0]

                if self.algo == 'SARSA':
                    delta = reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action]
                E[state][action] += 1

                #Update Q nad E
                self.Q += self.alpha*delta*E
                E = self.gamma*self.lmbd*E

                state = next_state
                action = next_action

            norms.append(np.linalg.norm(self.Q[1:22,1:11,:]))

        if log:
            print("Completed the total of {} episodes".format(n_episodes))
            print("The total time taken is {}s".format(time.time()-start))

        return self.Q, norms


from easy21 import Easy21, Action

env = Easy21()
env_name = 'Easy21'
TD = TDLambda(env, 2, (22,22))


Q = TD.train(n_episodes= 100000)

V = np.max(Q, axis = -1)
relevant_V = V[1:22,1:11]

import matplotlib.pyplot as plt

policy = np.argmax(Q, axis = -1)
plt.subplot(121)
plt.imshow(V)
plt.subplot(122)
plt.imshow(relevant_V)
plt.show()

from plot_cuts import *

matrix_surf(V[1:22,1:11], xlimits = [1,22], ylimits = [1,11], xlabel = 'Player Sum', ylabel='Dealer Showing', xticks=np.arange(1,22,2), yticks = np.arange(1,11))
