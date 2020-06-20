"""
Gambler's Problem

Gambler bets on:

H - wins twice he staked on the bet
T - looses what he staked on the bet

Game ends when he reaches 100$

action = {0, 1, ... min(s, 100-s)}
"""
import numpy as np
import matplotlib.pyplot as plt

p_head = 0.4
goal = 100

value_function = np.zeros(goal)
optimal_policy = np.zeros(goal, dtype = int)

def get_actions(state):
    global goal

    return list(range(min(state, goal - state) + 1))

def value_iteration(gamma = 0.9, delta = 10**(-3)):
    global prob_head, goal

    max_improvement = delta + 1

    while max_improvement > delta:
        max_improvement = 0

        for state in range(goal):
            best_improvement = -1
            actions = get_actions(state)
            old_state_value = value_function[state]

            for action in actions:
                #Betting amount action
                 loosing_state = state - action
                 winning_state = state + action

                 if loosing_state == 0:
                     loosing_value = 0
                 else:
                     loosing_value = value_function[loosing_state]

                 if winning_state == goal:
                     winning_value = 1
                     reward = 1
                 else:
                     winning_value = value_function[winning_state]
                 reward = 0

                 new_state_value = p_head*(gamma*winning_value + reward) + gamma*(1-p_head)*loosing_value

                 if new_state_value >= value_function[state]:
                     value_function[state] = new_state_value
                     improvement = abs(new_state_value - old_state_value)

                     if improvement > max_improvement:
                         max_improvement = improvement

def get_optimal_policy(gamma = 0.9):
    global prob_head, goal

    for state in range(goal):

        best_action_value = -1
        best_action = 1
        actions = get_actions(state)

        for action in actions:
            #Betting amount action
             loosing_state = state - action
             winning_state = state + action

             if loosing_state == 0:
                 loosing_value = 0
             else:
                 loosing_value = value_function[loosing_state]

             if winning_state == goal:
                 winning_value = 1
                 reward = 1
             else:
                 winning_value = value_function[winning_state]
             reward = 0

             new_state_value = p_head*(gamma*winning_value + reward) + gamma*(1-p_head)*loosing_value

             if new_state_value > best_action_value:
                 best_action_value = new_state_value
                 best_action = action

        optimal_policy[state] = best_action
value_iteration(gamma = 1, delta = 10**(-10))
get_optimal_policy()

plt.subplot(1,2,1)
plt.plot(value_function)
plt.subplot(1,2,2)
plt.plot(optimal_policy)
plt.show()
