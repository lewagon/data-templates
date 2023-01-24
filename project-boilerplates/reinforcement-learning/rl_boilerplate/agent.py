"""
Agent module.
"""

import random
import torch
import torch.nn
import tensorflow as tf

from rl_boilerplate import network
from rl_boilerplate.config import CFG


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


class DQNAgent_pt(Agent):
    """
    A basic pytorch Deep Q-learning agent.
    """

    def __init__(self, x_dim, y_dim):
        self.net = network.DQN_pt(x_dim, y_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """
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

class DQNAgent_tf(Agent):
    """
    A basic tensorflow Deep Q-learning agent.
    """

    def __init__(self, x_dim, y_dim):
        self.net = network.DQN_tf(x_dim, y_dim)
        self.opt = tf.optimizers.Adam(learning_rate=0.0001)

    def set(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """

        obs_new = obs_new.reshape(1, -1)

        with tf.GradientTape() as tape:

            # We get the network output
            out = self.net(obs_new)[0, act]

            # We compute the target
            exp = rwd + CFG.gamma * tf.reduce_max(self.net(obs_new))

            # Compute the loss
            loss = tf.square(exp - out)
            print(loss)

        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))

    def get(self, obs_new, act_space):
        """
        Run an epsilon-greedy policy for next actino selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon:
            return act_space.sample()
        # Else, return action with highest value
        with torch.no_grad():
            return tf.argmax(self.net(obs_new.reshape(1, -1)), axis=1).numpy()[0]
