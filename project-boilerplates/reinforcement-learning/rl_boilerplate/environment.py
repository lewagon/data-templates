"""
Environment module.

This module contains the RL environment. We provide a gym setup by default, which can easily be replaced by other packages such as pettingzoo. Fundamentally, this module is used to simulate the environment and generate (s, a, r, s') tuples for the agent to learn from.
"""

import gym

from config import CFG


def get_env():
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """
    # We use the LunarLander env. Other environments are available.
    return gym.make("LunarLander-v2")


def run_env(env, agt):
    """
    Run a given environment with a given agent.
    """

    obs_old, info = env.reset(seed=CFG.rnd_seed, return_info=True)

    # We get the action space.
    act_space = env.action_space

    for _ in range(1000):

        # We can visually render the learning environment. We disable it for performance.
        # env.render()

        # We request an action from the agent.
        act = agt.get(obs_old, act_space)

        # We apply the action on the environment.
        obs_new, rwd, end, _ = env.step(act)

        # We perform a learning step.
        agt.set(obs_old, act, rwd, obs_new)

        # Update latest observation
        obs_old = obs_new

        if end:
            obs_end, info = env.reset(return_info=True)

    env.close()
