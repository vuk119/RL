def monte_carlo_control(env, action_space = [0,1], n_0 = 100, n_of_episodes = 1000):
    #First axis is player state, second axis is dealer state and third axis is action
    Q = np.zeros((22, 22, 2))
    policy = lambda player_state, dealer_state, epsilon : np.argmax(Q[player_state, dealer_state, :]) if np.random.random() > epsilon/2  \
                                                            else np.argmin(Q[player_state, dealer_state, :])

    returns = np.zeros((22,22,2))
    N_state = np.zeros((22,22), dtype = int)
    N_state_action = np.zeros((22,22,2), dtype = int)

    start_time = time.time()

    for _ in range(n_of_episodes):

        if _%100000==0:
            print("Episode number: {}".format(_))
            print("Total time passed: {}s".format(time.time()-start_time))

        player_state, dealer_state = env.reset()

        #Record the data
        seen_states = [(player_state, dealer_state)]
        seen_actions = []

        #Ensures first-visit MC control
        visited_states = np.zeros((22,22,2), dtype = bool)

        done = False

        #Sample an Episode
        while not done:

            epsilon = n_0 / (n_0 + N_state[player_state, dealer_state])

            action = policy(player_state, dealer_state, epsilon)

            N_state_action[player_state,dealer_state,action] += 1
            N_state[player_state, dealer_state] += 1

            (player_state, dealer_state), reward, done = env.step(action)
            if visited_states[player_state, dealer_state, action] == 0:
                seen_states.append((player_state, dealer_state))
                seen_actions.append(action)
                visited_states[player_state, dealer_state, action] = 1


        #Update Q table
        for i in range(len(seen_states)-1):
            (player_state, dealer_state) = seen_states[i]
            action = seen_actions[i]

            returns[player_state,dealer_state,action]+=reward

            Q[player_state,dealer_state,action] = returns[player_state,dealer_state,action]/N_state_action[player_state,dealer_state,action]

    return Q


Q = monte_carlo_control(Easy21(), n_of_episodes = 5000)
V = np.max(Q, axis = -1)
relevant_V = V[1:22,1:11]

policy = np.argmax(Q, axis = -1)
plt.subplot(121)
plt.imshow(V)
plt.subplot(122)
plt.imshow(relevant_V)
plt.show()


from plot_cuts import *

matrix_surf(V[1:22,1:11], xlimits = [1,22], ylimits = [1,11], xlabel = 'Player Sum', ylabel='Dealer Showing', xticks=np.arange(1,22,2), yticks = np.arange(1,11))

plt.imshow(policy[:,1:11])
plt.show()
