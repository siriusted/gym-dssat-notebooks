import gym
import gym_dssat_pdi
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# First, we instantiate the environment. Here, we will use the fertilization mode..
env_args = {
  'run_dssat_location': '/opt/dssat_pdi/run_dssat',
  'mode': 'fertilization',
  'seed': 123,
  'random_weather': True,
}

env = gym.make('GymDssatPdi-v0', **env_args)

# helper functions to define hardcoded policies
def null_policy(obs):
    '''
    Do not apply any fertilizer
    '''
    return { "anfer": 0}

def expert_policy(obs):
    '''
    Return fertilization amount based on internal map
    and day after planting feature in observation.
    '''
    fertilization_map = {
        40: 27,
        45: 35,
        80: 54,
    }

    amount = fertilization_map.get(obs['dap'], 0)
    
    return { "anfer": amount }

def evaluate(policy, n_episodes=10):
    '''evaluates a policy for n_episodes episodes'''
    returns = [None] * n_episodes

    for episode in range(n_episodes):
        done = False
        obs = env.reset()
        ep_return = 0

        while not done:
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            if reward is not None:
                ep_return += reward

        returns[episode] = ep_return

    return returns

# Evaluate null policy
print('Evaluating null policy...')
null_returns = evaluate(null_policy)
print('Done')

# Evaluate expert policy
print('Evaluating expert policy...')
expert_returns = evaluate(expert_policy)
print('Done')

# write results to a file to be loaded for display
data = [('null', null_returns), ('expert', expert_returns)]
with open("results.pkl", "wb") as result_file:
    pickle.dump(data, result_file)

# Cleanup
env.close()
