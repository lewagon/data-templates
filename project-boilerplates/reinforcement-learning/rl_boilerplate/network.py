"""
Neural network module.

This module defines architectures used by reinforcement learning agents.
"""

import tensorflow as tf
import torch
import torch.nn


class DQN_pt(torch.nn.Module):
    """
    PyTorch implementation of a Deep Q-Network with 3 linear layers.
    x_dim refers to the number of dimensions to pass as input
    y_dim refers to the action space of the agent
    """

    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(x_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, y_dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        return self.net(obs)

class DQN_tf(tf.keras.Model):
    """
    Tensorflow implementation of a Deep Q-Network with 3 linear layers.
    x_dim refers to the number of dimensions to pass as input
    y_dim refers to the action space of the agent
    """

    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation="relu", input_shape=(x_dim,))
        self.layer2 = tf.keras.layers.Dense(128, activation="relu")
        self.layer3 = tf.keras.layers.Dense(y_dim, activation="relu")

    def call(self, obs):
        x = self.layer1(obs)
        x = self.layer2(x)
        return self.layer3(x)
