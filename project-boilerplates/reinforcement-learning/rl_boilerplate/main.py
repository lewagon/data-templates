from rl_boilerplate import agent, environment
from rl_boilerplate.config import CFG
from tqdm import tqdm

# We initialize our configuration class
print("initialize config:")
CFG.init("", epsilon=0.8)
print(CFG.__dict__)

# We create an agent.
#agt = agent.DQNAgent_tf(8, 4)
agt = agent.DQNAgent_pt(8, 4)

# Run a learning process
for i in tqdm(range(1, 1000)):
    print(f"\n 💫 Run number: {i}\n ")
    env = environment.get_env()
    environment.run_env(env, agt)

    if i % 10 == 0:
        print("Reduce exploration rate:")
        CFG.init("", epsilon=CFG.epsilon * 0.5)
        print(CFG.__dict__)
