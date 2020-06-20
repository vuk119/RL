import gym

from DDQN import DDQN

env = gym.make('CartPole-v0')
filepath = './out/128-ddqn-CartPole-v0-1592501409/checkpoints/n0'
render = True

observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

ddqn = DDQN(observation_space_size, action_space_size, play_mode=True)
ddqn.load_checkpoint(filepath)

state = env.reset()
done = False
total_reward = 0

while done is not True:
    action = ddqn.get_action(state, epsilon=0)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    if render is True:
        env.render()
env.close()

print("The total reward is:", total_reward)
