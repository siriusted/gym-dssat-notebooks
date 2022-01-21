import gym
import gym_dssat_pdi
import seaborn as sns
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

def plot_results(labels, returns):
    data_dict = {}
    for label, data in zip(labels, returns):
        data_dict[label] = data
    df = pd.DataFrame(data_dict)
    
    ax = sns.boxplot(data=df)
    ax.set_xlabel("policy")
    ax.set_ylabel("evaluation output")
    plt.savefig('results.pdf')
    plt.show()


# Evaluate null policy
null_returns = evaluate(null_policy)

# Evaluate expert policy
expert_returns = evaluate(expert_policy)

# Display results and save a copy as `results.pdf` in the current working directory
labels = ['null', 'expert']
returns = [null_returns, expert_returns]
plot_results(labels, returns)
