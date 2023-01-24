"""
Neural network module.

This module defines architectures used by reinforcement learning agents.
"""

import torch
import torch.nn


class DQN(torch.nn.Module):
    """
    A simple Deep Q-Network with 3 linear layers.
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
