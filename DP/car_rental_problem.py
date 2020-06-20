"""
                           Car Rental Problem

Two renting car locations.

-If a customer comes and rents a car the reward is 10$.
-If he is out of cars the business is lost.
-Cars are avaliable a day after they are returned
-Number of requested cars ~ Poisson(3)
-Number of returned cars  ~ Poisson(4)
-No more than 20 cars at each location. (Any extra cars just dissappear)
-Each night you can move 0,1,2,3,4 or 5 cars between the two locations
-The cost of moving a car is 2$
-Gamma = 0.9


states: (0,0) -> (20,20)
actions: [-5,-4,-3,-2,-1,0,1,2,3,4,5] - How many cars we move from location 1 to location 2

The value function of states (0,x) and (x,0) is 0

"""

import time
import os
import pickle

import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

moving_cost = 2
rent_price = 10
max_cars = 20
request_mean_1 = 3
request_mean_2 = 4
return_mean_1 = 3
return_mean_2 = 2
compute = True


def compute_dynamics(init_state):
    global max_cars, request_mean_1, request_mean_2, return_mean_1, return_mean_2, rent_price, moving_cost

    i, j = init_state

    transition_probabilities_and_rewards = {}



    #HANDLE REQUESTS
    #CASE 1: LOST BUSSINES
    prob_lost_1 = 1 - stats.poisson.cdf(k=i, mu = request_mean_1)
    prob_lost_2 = 1 - stats.poisson.cdf(k=j, mu = request_mean_2)

    prob_lost = prob_lost_1 + prob_lost_2 - prob_lost_1*prob_lost_2

    #CASE 2: NOT LOST
    #LOOP OVER VALID RENTS
    for req_1 in range(i + 1):
        for req_2 in range(j + 1):

            req_prob = stats.poisson.pmf(k = req_1, mu = request_mean_1) * stats.poisson.pmf(k = req_2, mu = request_mean_2)
            reward = (req_1 + req_2)*rent_price

            #LOOP OVER RETURNS
            for ret_1 in range(max_cars + 1 - (i-req_1)):

                probb = 0
                for ret_2 in range(max_cars + 1 - (j-req_2)):
                    #ADD IFS
                    if ret_1 == max_cars - (i-req_1):
                        ret_1_prob = 1 - stats.poisson.cdf(k = ret_1 - 1, mu = return_mean_1)
                    else:
                        ret_1_prob = stats.poisson.pmf(k = ret_1, mu = return_mean_1)
                    if ret_2 == max_cars - (j-req_2):
                        ret_2_prob = 1 - stats.poisson.cdf(k = ret_2 - 1, mu = return_mean_2)
                    else:
                        ret_2_prob = stats.poisson.pmf(k = ret_2, mu = return_mean_2)

                    final_state = (i - req_1 + ret_1, j - req_2 + ret_2)

                    ret_prob =  ret_1_prob * ret_2_prob
                    probb += ret_2_prob

                    if final_state not in transition_probabilities_and_rewards:
                        transition_probabilities_and_rewards[final_state] = []
                    transition_probabilities_and_rewards[final_state].append((req_prob*ret_prob, reward))


    transition_probabilities_and_expected_rewards = {}

    transition_probabilities = np.zeros((max_cars+1, max_cars+1))
    expected_rewards = np.zeros((max_cars+1, max_cars+1))

    for state in transition_probabilities_and_rewards:
        total_prob = 0
        expected_reward = 0

        for prob, reward in transition_probabilities_and_rewards[state]:
            total_prob += prob
            expected_reward += prob*reward


        transition_probabilities[state[0], state[1]] = total_prob
        expected_rewards[state[0], state[1]] = expected_reward / total_prob

    if 1 - np.sum(np.sum(transition_probabilities)) - prob_lost > 0.01:
        print("ATTENTION")
        print("Transition probabilities for this state do not add up to 1!")
        print(init_state, "{:.2f}".format(100*(1 - np.sum(np.sum(transition_probabilities)) - prob_lost)))
    return transition_probabilities, expected_rewards, prob_lost

def compute_dynamics_one_it(i):
    global max_cars, transition_probabilities, expected_rewards, lost_probabilities

    start = time.time()
    for j in range(max_cars+1):
            transition_probabilities[i,j,:,:], expected_rewards[i,j,:,:], lost_probabilities[i,j] = compute_dynamics((i,j))
            print(i,j)
    print(i, time.time() - start,'s')

def compute_full_dynamics_in_parallel():
    global max_cars, transition_probabilities, expected_rewards, lost_probabilities

    pool_size = multiprocessing.cpu_count()-1
    #pool_size = 4
    pool = Pool(pool_size)

    start = time.time()
    for i in range(max_cars+1):
        pool.apply_async(compute_dynamics_one_it, (i,))

    pool.close()
    pool.join()

    with open('./pymdp/saved_variab les/transition_probabilities.pkl', 'wb') as f:
        pickle.dump(transition_probabilities, f)
    with open('./pymdp/saved_variables/expected_rewards.pkl', 'wb') as f:
        pickle.dump(expected_rewards, f)
    with open('./pymdp/saved_variables/lost_probabilities.pkl', 'wb') as f:
        pickle.dump(lost_probabilities, f)

def compute_full_dynamics():
    global max_cars, transition_probabilities, expected_rewards, lost_probabilities

    start = time.time()
    for i in range(max_cars+1):
        print(i, time.time() - start,'s')
        for j in range(max_cars+1):
                transition_probabilities[i,j,:,:], expected_rewards[i,j,:,:], lost_probabilities[i,j] = compute_dynamics((i,j))


    with open('./pymdp/saved_variables/transition_probabilities.pkl', 'wb') as f:
        pickle.dump(transition_probabilities, f)
    with open('./pymdp/saved_variables/expected_rewards.pkl', 'wb') as f:
        pickle.dump(expected_rewards, f)
    with open('./pymdp/saved_variables/lost_probabilities.pkl', 'wb') as f:
        pickle.dump(lost_probabilities, f)

def load_dynamics():
    global transition_probabilities, expected_rewards, lost_probabilities

    with open('./saved_variables/transition_probabilities.pkl', 'rb') as f:
        transition_probabilities = pickle.load(f)
    with open('./saved_variables/expected_rewards.pkl', 'rb') as f:
        expected_rewards = pickle.load(f)
    with open('./saved_variables/lost_probabilities.pkl', 'rb') as f:
        lost_probabilities = pickle.load(f)

def value_iteration(gamma = 0.9, delta = 10**(-3)):
    global max_cars, request_mean_1, request_mean_2, return_mean_1, return_mean_2, rent_price, moving_cost, transition_probabilities, expected_rewards, optimal_policy

    max_improvement = 10

    v = np.zeros((max_cars + 1, max_cars + 1))

    while max_improvement > delta:
        #print('HI', max_improvement)
        max_improvement = 0

        #LOOP OVER STATES
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                best_value = -1

                #LOOP OVER ACTIONS
                for a in actions:
                    new_i = i - a
                    new_j = j + a
                    if new_i>max_cars or new_i<0 or new_j>max_cars or new_j<0:
                        continue

                    action_value = np.sum(np.sum(np.multiply(transition_probabilities[new_i, new_j], gamma*v + expected_rewards[new_i,new_j]))) - abs(a)*moving_cost

                    if action_value > best_value:
                        best_value = action_value

                improvement = abs(v[i,j] - best_value)
                v[i,j] = best_value

                if improvement > max_improvement:
                    max_improvement = improvement

    return v

def get_optimal_policy(v, gamma = 0.9):
    optimal_policy = np.zeros((max_cars + 1, max_cars + 1), dtype = int)

    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            best_value = -1

            #LOOP OVER ACTIONS
            for a in actions:
                new_i = i - a
                new_j = j + a
                if new_i>max_cars or new_i<0 or new_j>max_cars or new_j<0:
                    continue

                action_value = np.sum(np.sum(np.multiply(transition_probabilities[new_i, new_j], gamma*v + expected_rewards[new_i,new_j]))) - abs(a)*moving_cost

                if action_value >= best_value:
                    best_value = action_value
                    optimal_policy[i,j] = a

    return optimal_policy

def policy_evaluation(policy, gamma = 0.9, delta = 10**(-3)):
    global max_cars, request_mean_1, request_mean_2, return_mean_1, return_mean_2, rent_price, moving_cost, v, transition_probabilities, expected_rewards

    v = np.zeros((max_cars + 1, max_cars + 1))

    max_improvement = delta + 1

    while max_improvement > delta:
        max_improvement = 0

        #LOOP OVER STATES
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):

                action = policy[i,j]
                new_i = i - action
                new_j = j + action

                new_value = np.sum(np.sum(np.multiply(transition_probabilities[new_i, new_j], gamma*v + expected_rewards[new_i,new_j]))) - abs(action)*moving_cost

                improvement = abs(new_value - v[i,j])

                if improvement > max_improvement:
                    max_improvement = improvement

                v[i,j] = new_value

    return v

def policy_improvement(policy, v, gamma = 0.9, delta = 10**(-3)):
    global max_cars, request_mean_1, request_mean_2, return_mean_1, return_mean_2, rent_price, moving_cost, transition_probabilities, expected_rewards

    policy_stable = True

    for i in range(max_cars + 1):
        for j in range(max_cars + 1):

            current_action = policy[i,j]
            new_i = i - current_action
            new_j = j + current_action
            current_action_value = np.sum(np.sum(np.multiply(transition_probabilities[new_i, new_j], gamma*v + expected_rewards[new_i,new_j]))) - abs(current_action)*moving_cost

            #LOOP OVER ACTIONS
            for a in actions:
                new_i = i - a
                new_j = j + a
                if new_i>max_cars or new_i<0 or new_j>max_cars or new_j<0:
                    continue

                action_value = np.sum(np.sum(np.multiply(transition_probabilities[new_i, new_j], gamma*v + expected_rewards[new_i,new_j]))) - abs(a)*moving_cost

                if action_value > current_action_value:
                    v[i,j] = action_value
                    policy[i,j] = a
                    policy_stable = False

                    current_action_value = action_value

    return policy, policy_stable

def policy_iteration(k = 3, gamma = 0.9, delta = 10**(-3)):


    v = np.zeros((max_cars + 1, max_cars + 1))
    policy = np.zeros((max_cars + 1, max_cars + 1), dtype = int)

    policy_stable = False

    while policy_stable is False:
        v = policy_evaluation(policy, gamma, delta)
        policy, policy_stable = policy_improvement(policy, v, gamma, delta)

    return v, policy

def plot_value_function3D():
    X_arr = []
    Y_arr = []
    Z_arr = []

    for i in range(max_cars+1):
        for j in range(max_cars+1):
            X_arr.append(i)
            Y_arr.append(j)
            Z_arr.append(v[i,j])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_arr, Y_arr, Z_arr)
    plt.show()

def plot_optimal_policy3D():
    X_arr = []
    Y_arr = []
    Z_arr = []

    for i in range(max_cars+1):
        for j in range(max_cars+1):
            X_arr.append(i)
            Y_arr.append(j)
            Z_arr.append(optimal_policy[i,j])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_arr, Y_arr, Z_arr)
    plt.show()


value_function = np.zeros((max_cars + 1, max_cars + 1))
actions = [a for a in range(-5, 6)]

transition_probabilities = np.zeros((max_cars + 1, max_cars + 1, max_cars + 1, max_cars + 1))
expected_rewards = np.zeros((max_cars + 1, max_cars + 1, max_cars + 1, max_cars + 1))
lost_probabilities = np.zeros((max_cars + 1, max_cars + 1))

#compute_full_dynamics_in_parallel()
load_dynamics()

value_function = value_iteration()
optimal_policy = get_optimal_policy(value_function)

value_function1, optimal_policy1 = policy_iteration()


#UNIT TEST FOR POLICY EVALUATION
#print(v is policy_evaluation(optimal_policy))
#print(np.sum((np.sum(v - policy_evaluation(optimal_policy))>10**(-3))))

#Compare Algorithms
print("Value functions differ in {} states".format(np.sum((np.sum(value_function1 - value_function>10**(-3))))))
print('Are value functions the same objects?', value_function is value_function1)

print('Are obtained policies the same in value?',(optimal_policy == optimal_policy1).all())
print('Are obtained policies the same objects?', optimal_policy is optimal_policy1)

assert False
