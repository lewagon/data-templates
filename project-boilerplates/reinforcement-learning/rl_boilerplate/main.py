import agent
import environment

from config import CFG

# We initialize our configuration class
CFG.init("", rnd_seed=22)

# We create an agent. State and action spaces are hardcoded here.
agt = agent.DQNAgent(8, 4)

# Run a learning process
for _ in range(1000):
    env = environment.get_env()
    environment.run_env(env, agt)
