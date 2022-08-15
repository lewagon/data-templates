This is a **boilerplate** repo for a reinforcement learning (RL) project.

This directory provides an example repository structure for RL projects using pytorch. This template provides a generic agent using the [deep Q-learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) algorithm as well as an agent playing random actions for baseline performance. The DQN architecture is in itw own class and is hot-swappable with other potential architectures. A sample environment using [OpenAi's gym](https://github.com/openai/gym) and a generic control loop is also provided.

Note that since RL projects are rarely data-centric, and data has to be generated on-the-fly, requirements are likely to differ from standard ML projects.

# Detailed package workflow

This boilerplate package contains multiple modules:

- `main.py` is the entry point of the package. It defines the agent and environment to use.
- `environment.py` defines environment-side setup and execution utilities. It uses the gym package for demonstration purposes.
- `agent.py` defines multiple types of learning agent. We have included a random agent and deep Q-learning agent for demonstration purposes.
- `config.py` defines a singleton class used for storing simulation parameters. This class is globally available in all packages (through the `CFG` variable). It has to be initialized once (see module documentation).
- `network.py` defines the neural network used by the DQN agent.
