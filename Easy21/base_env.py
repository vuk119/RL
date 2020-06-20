from abc import ABCMeta, abstractmethod

class BaseEnv(object):
    __metaclass__=ABCMeta

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self,action):
        raise NotImplementedError

    def get_state_space_shape(self):
        return self.state_space_shape

    def get_action_space_shape(self):
        return self.action_space_shape

    def get_action_space(self):
        return self.action_space
