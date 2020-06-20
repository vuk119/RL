import numpy as np
import matplotlib.pyplot as plt

class Environment():
    def actions(self):
        raise NotImplementedError('Inheriting classes must override actions.')

    def step(self):
        raise NotImplementedError('Inheriting classes must override step')

class BanditEnv():

    def __init__(self, choices):
        """ here choices is a list of distributions """
        """ each element is a tuple (name,(parameters)) """
        """ avaliable distributions: """
        """ exponential - parameter is lambda """
        """ poisson - parameter is lambda """
        """ uniform - parameters are interval limits """
        self.num_actions=len(choices)
        self.choices=choices

    def step(self, action):
        dist=self.choices[action][0]
        parameters=self.choices[action][1]

        if dist=="exponential":
            return np.random.exponential(parameters)
        if dist=="poisson":
            return np.random.poisson(parameters)
        if dist=="uniform":
            return np.random.uniform(parameters[0],parameters[1])
        if dist=="normal":
            return np.random.normal(parameters[0],parameters[1])

    def get_optimal_average(self):
        max_average=-float('inf')

        for dist,parameters in self.choices:
            if dist=="exponential":
                average=parameters
            if dist=="poisson":
                average=parameters
            if dist=="uniform":
                average=(parameters[0]+parameters[1])/2
            if dist=="normal":
                average=parameters[0]

            if average>max_average:
                max_average=average

        return max_average

class Agent():
    def __init__(self, num_actions):
        self.num_actions=num_actions
    def act(self):
        raise NotImplementedError
    def update(self):
        raise NotImplementedError
class RandomAgent(Agent):
    def __init__(self,num_actions):
        super(RandomAgent,self).__init__(num_actions)
        self.rewards=[]
    def act(self):
        return np.random.randint(0,self.num_actions)
    def update(self, reward):
        self.rewards.append(reward)
class EpsilonGreedyAgent(Agent):

    def __init__(self, num_actions, epsilon=0.01):
        super(EpsilonGreedyAgent, self).__init__(num_actions)
        self.rewards=[]
        self.averages=[0]*num_actions
        self.frequencies=[0]*num_actions
        self.epsilon=epsilon
    def act(self):
        if np.random.rand()>self.epsilon:
            self.last_action=np.argmax(self.averages)
        else:
            self.last_action=np.random.choice(range(self.num_actions))

        self.frequencies[self.last_action]+=1
        return self.last_action
    def update(self, reward):
        self.rewards.append(reward)
        """ maintaining the average according to the: """
        """ Q_{n+1}=\frac{n*Q_n+q_{n+1}}{n+1} """
        self.averages[self.last_action]=((self.frequencies[self.last_action]-1)*self.averages[self.last_action]+reward) \
                                                                               /self.frequencies[self.last_action]
class OptimisticAgent(Agent):

    def __init__(self, num_actions, optimistic_average=10):
        super(OptimisticAgent, self).__init__(num_actions)
        self.rewards=[]
        self.averages=[optimistic_average]*num_actions
        self.frequencies=[0]*num_actions

    def act(self):
        self.last_action=np.argmax(self.averages)
        self.frequencies[self.last_action]+=1
        return self.last_action
    def update(self, reward):
        self.rewards.append(reward)
        """ maintaining the average according to the: """
        """ Q_{n+1}=\frac{n*Q_n+q_{n+1}}{n+1} """
        self.averages[self.last_action]=((self.frequencies[self.last_action]-1)*self.averages[self.last_action]+reward) \
                                                                               /self.frequencies[self.last_action]
class UCBAgent(Agent):

    def __init__(self, num_actions, c=2):
        super(UCBAgent, self).__init__(num_actions)
        self.rewards=[]
        self.averages=[0]*num_actions
        self.frequencies=[0]*num_actions
        self.c=c

    def act(self):
        """UPPER CONFIDENCE BOUND """
        """this algorithm selects the action that has the maximum value of the function"""
        """average+c*sqrt(ln(t)/frequency)"""

        max_confidence=-float('inf')
        self.last_action=0

        for action in range(len(self.averages)):
            current_confidence=self.averages[action]+ \
                               self.c*np.sqrt(np.log(len(self.rewards)+1)/(self.frequencies[action]+1))
            if current_confidence > max_confidence:
                max_confidence=current_confidence
                self.last_action=action

        self.frequencies[self.last_action]+=1

        return self.last_action
    def update(self, reward):
        self.rewards.append(reward)
        """ maintaining the average according to the: """
        """ Q_{n+1}=\frac{n*Q_n+q_{n+1}}{n+1} """
        self.averages[self.last_action]=((self.frequencies[self.last_action]-1)*self.averages[self.last_action]+reward) \
                                                                               /self.frequencies[self.last_action]
class GradientAgent(Agent):

    def __init__(self, num_actions, alpha=0.1):
        super(GradientAgent, self).__init__(num_actions)
        self.rewards=[]
        self.preferences=[0]*num_actions
        self.alpha=alpha
        self.num_actions=num_actions

    def act(self):
        probabilities=np.exp(self.preferences)
        probabilities/=np.sum(probabilities)
        self.last_action=np.random.choice(np.arange(self.num_actions),p=probabilities)
        return self.last_action

    def update(self, reward):
        self.rewards.append(reward)
        average_reward=np.mean(self.rewards)

        probabilities=np.exp(self.preferences)
        probabilities/=np.sum(probabilities)

        for i in range(len(self.preferences)):
            if i==self.last_action:
                self.preferences[i]+=self.alpha*(reward-average_reward)*(1-probabilities[i])
            else:
                self.preferences[i]-=self.alpha*(reward-average_reward)*probabilities[i]


def get_averages(Agent):
    net=0
    averages=[]

    for i in range(len(Agent.rewards)):
        net+=Agent.rewards[i]
        averages.append(net/(i+1))

    return averages
def get_regret(Agent,optimal_average):
    net=0
    regrets=[]

    for i in range(len(Agent.rewards)):
        net+=Agent.rewards[i]
        regrets.append(optimal_average*(i+1)-net)

    return regrets

def RandomAgentPlot():
    Bandit=BanditEnv([("uniform",[1,2]),("uniform",[3,4]),("uniform",[0,2])])
    Random=RandomAgent(3)
    N_steps=1000
    for i in range(N_steps):
        action=Random.act()
        Random.update(Bandit.step(action))

    fig=plt.figure(figsize=(30, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(121)
    plt.ylim([0,1.1])
    plt.ylabel("Average reward / Optimal reward []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.title("Average reward over time", fontsize=30)
    plt.plot(np.divide(get_averages(Random),Bandit.get_optimal_average()),label="Random Agent")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')

    plt.subplot(122)
    plt.title("Regret over time", fontsize=30)
    plt.ylabel("Regret []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.plot(get_regret(Random,Bandit.get_optimal_average()),label="Random Agent")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')

def GreedyPlot(N):
    Bandit=BanditEnv([("uniform",[1,2]),("uniform",[3,4]),("uniform",[0,2])])
    N_steps=N
    Epsilon001=EpsilonGreedyAgent(3,0.01)
    Epsilon005=EpsilonGreedyAgent(3,0.05)
    Epsilon01=EpsilonGreedyAgent(3,0.1)
    Optimistic=OptimisticAgent(3,5)
    for i in range(N_steps):
        action=Epsilon001.act()
        Epsilon001.update(Bandit.step(action))
    for i in range(N_steps):
        action=Epsilon01.act()
        Epsilon01.update(Bandit.step(action))
    for i in range(N_steps):
        action=Epsilon005.act()
        Epsilon005.update(Bandit.step(action))
    for i in range(N_steps):
        action=Optimistic.act()
        Optimistic.update(Bandit.step(action))

    fig=plt.figure(figsize=(30, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(121)
    plt.ylim([0,1.1])
    plt.ylabel("Average reward / Optimal reward []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.title("Average reward over time", fontsize=30)
    plt.plot(np.divide(get_averages(Epsilon001),Bandit.get_optimal_average()),label="epsilon=0.01")
    plt.plot(np.divide(get_averages(Epsilon005),Bandit.get_optimal_average()),label="epsilon=0.05")
    plt.plot(np.divide(get_averages(Epsilon01),Bandit.get_optimal_average()),label="epsilon=0.1")
    plt.plot(np.divide(get_averages(Optimistic),Bandit.get_optimal_average()),label="optimistic")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')

    plt.subplot(122)
    plt.title("Regret over time", fontsize=30)
    plt.ylabel("Regret []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.plot(get_regret(Epsilon001,Bandit.get_optimal_average()),label="epsilon=0.01")
    plt.plot(get_regret(Epsilon005,Bandit.get_optimal_average()),label="epsilon=0.05")
    plt.plot(get_regret(Epsilon01,Bandit.get_optimal_average()),label="epsilon=0.1")
    plt.plot(get_regret(Optimistic,Bandit.get_optimal_average()),label="optimistic")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')

def UCBGradientPlot(N):
    Bandit=BanditEnv([("uniform",[1,2]),("uniform",[3,4]),("uniform",[0,2])])
    N_steps=N
    UCB=UCBAgent(3,2)
    Gradient=GradientAgent(3,0.1)
    Epsilon01=EpsilonGreedyAgent(3,0.1)
    Optimistic=OptimisticAgent(3,5)
    for i in range(N_steps):
        action=UCB.act()
        UCB.update(Bandit.step(action))
    for i in range(N_steps):
        action=Gradient.act()
        Gradient.update(Bandit.step(action))
    for i in range(N_steps):
        action=Optimistic.act()
        Optimistic.update(Bandit.step(action))

    fig=plt.figure(figsize=(30, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(121)
    plt.ylim([0,1.1])
    plt.ylabel("Average reward / Optimal reward []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.title("Average reward over time", fontsize=30)
    plt.plot(np.divide(get_averages(UCB),Bandit.get_optimal_average()),label="UCB")
    plt.plot(np.divide(get_averages(Gradient),Bandit.get_optimal_average()),label="gradient")
    plt.plot(np.divide(get_averages(Optimistic),Bandit.get_optimal_average()),label="optimistic")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')

    plt.subplot(122)
    plt.title("Regret over time", fontsize=30)
    plt.ylabel("Regret []", fontsize=20)
    plt.xlabel("Timesteps []", fontsize=20)
    plt.plot(get_regret(UCB,Bandit.get_optimal_average()),label="UCB")
    plt.plot(get_regret(Gradient,Bandit.get_optimal_average()),label="gradient")
    plt.plot(get_regret(Optimistic,Bandit.get_optimal_average()),label="optimistic")
    plt.legend(loc=[0.70,0.70], fontsize='xx-large')
