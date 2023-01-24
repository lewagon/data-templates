"""
Agent module.
"""

import random
import torch
import torch.nn

import network
from config import CFG


class Agent:
    """
    A learning agent parent class.
    """

    def __init__(self):
        pass

    def set(self):
        """
        Make the agent learn from a (s, a, r, s') tuple.
        """
        raise NotImplementedError

    def get(self):
        """
        Request a next action from the agent.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    A random playing agent class.
    """

    def set(self, obs_old, act, rwd, obs_new):
        """
        A random agent doesn't learn.
        """
        return

    def get(self, obs_new, act_space):
        """
        Simply return a random action.
        """
        return act_space.sample()


class DQNAgent(Agent):
    """
    A basic Deep Q-learning agent.
    """

    def __init__(self, x_dim, y_dim):
        self.net = network.DQN(x_dim, y_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """

        obs_old = torch.tensor(obs_old)
        obs_new = torch.tensor(obs_new)

        # We get the network output
        out = self.net(torch.tensor(obs_new))[act]

        # We compute the target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(obs_new).max()

        # Compute the loss
        loss = torch.square(exp - out)

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()

    def get(self, obs_new, act_space):
        """
        Run an epsilon-greedy policy for next actino selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon:
            return act_space.sample()
        # Else, return action with highest value
        with torch.no_grad():
            # Get the values of all possible actions
            val = self.net(torch.tensor(obs_new))
            # Choose the highest-values action
            return torch.argmax(val).numpy()
