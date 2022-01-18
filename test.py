import gym
import gym_dssat_pdi

env_args = {
    'run_dssat_location': '/opt/dssat_pdi/run_dssat',  # assuming (modified) DSSAT has been installed in /opt/dssat_pdi
    'log_saving_path': './logs/dssat-pdi.log',  # if you want to save DSSAT outputs for inspection
    # 'mode': 'irrigation',  # you can choose one of those 3 modes
    # 'mode': 'fertilization',
    'mode': 'all',
    'seed': 123456,
    'random_weather': True,  # if you want stochastic weather
}

env = gym.make('GymDssatPdi-v0', **env_args)

done = False
obs = env.reset()

print('Running environment for one episode using a random policy...')
print(f'\nInitial observation is {obs}')
total_return = 0

while not done:
  obs, reward, done, info = env.step(env.action_space.sample())
  total_return += reward if reward is not None else 0
  
print(f'Final observation is {obs}')
print(f'Total return over one episode with random policy is: {total_return}')
