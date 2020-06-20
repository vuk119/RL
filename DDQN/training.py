from itertools import count

import matplotlib.pyplot as plt
import gym

from DDQN import DDQN

name = '128-ddqn'
env_name = 'CartPole-v0'
env = gym.make(env_name)

observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

ddqn = DDQN(observation_space_size, action_space_size, name, env_name)

n_episodes = 200
rewards = []

for i_episode in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    for t in count():
        action = ddqn.get_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        ddqn.update_memory([state, action, reward, next_state, done])

        state = next_state

        ddqn.update_from_batch()

        if t % 10 == 0:
            ddqn.update_target_net()

        if done:
            rewards.append(episode_reward)
            break

ddqn.save_checkpoint()
plt.plot(rewards)
plt.show()
