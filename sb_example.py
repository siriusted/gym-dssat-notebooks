import gym
import gym_dssat_pdi
import pickle
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    low, high = action_space_limits
    return 2.0 * ((action - low) / (high - low)) - 1.0

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    low, high = action_space_limits
    return low + (0.5 * (action + 1.0) * (high - low))

# Wrapper for easy and uniform interfacing with SB3
class GymDssatWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymDssatWrapper, self).__init__(env)

        self.action_low, self.action_high = self._get_action_space_bounds()

        # using a normalized action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype="float32")

        # using a vector representation of observations to allow
        # easily using SB3 MlpPolicy
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=np.inf,
                                                shape=env.observation_dict_to_array(
                                                    env.observation).shape,
                                                dtype="float32"
                                                )
        
        # to avoid annoying problem with Monitor when episodes end and things are None
        self.last_info = {}
        self.last_obs = None

    def _get_action_space_bounds(self):
        box = self.env.action_space['anfer']
        return box.low, box.high

    def _format_action(self, action):
        return { 'anfer': action[0] }

    def _format_observation(self, observation):
        return self.env.observation_dict_to_array(observation)

    def reset(self):
        return self._format_observation(self.env.reset())
        

    def step(self, action):
        # Rescale action from [-1, 1] to original action space interval
        denormalized_action = denormalize_action((self.action_low, self.action_high), action)
        formatted_action = self._format_action(denormalized_action)
        obs, reward, done, info = self.env.step(formatted_action)

        # handle `None`s in obs, reward, and info on done step
        if done:
            obs, reward, info = self.last_obs, 0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self._format_observation(obs)
        return formatted_observation, reward, done, info

    def close(self):
        return self.env.close()

    def seed(self, seed):
        self.env.set_seed(seed)

    def __del__(self):
        self.close()

# Create environment
env_args = {
  'run_dssat_location': '/opt/dssat_pdi/run_dssat',
  'mode': 'fertilization',
  'seed': 123,
  'random_weather': True,
}

env = GymDssatWrapper(gym.make('GymDssatPdi-v0', **env_args))

# Training arguments for PPO agent
ppo_args = {
    'batch_size': 128,
    'n_steps': 256,
    'gamma': 0.95,
    'learning_rate': 0.003,
    'clip_range': 0.1,
    'n_epochs': 20,
    'policy_kwargs': dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=torch.nn.Tanh,
        ortho_init=False,
    ),
    'seed': 123,
}

# Create the agent
ppo_agent = PPO('MlpPolicy', env, **ppo_args)

# Train for 40k timesteps
print('Training PPO agent...')
ppo_agent.learn(total_timesteps=40_000)
print('Training done')

# Baseline agents for comparison
class NullAgent:
    """
    Agent always choosing to do no fertilization
    """
    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, deterministic=None):
        action = normalize_action((self.env.action_low, self.env.action_high), [0])
        return np.array([action], dtype=np.float32), obs


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """
    fertilization_dic = {
        40: 27,
        45: 35,
        80: 54,
    }

    def __init__(self, env, normalize_action=False, fertilization_dic=None):
        self.env = env
        self.normalize_action = normalize_action

    def _policy(self, obs):
        dap = int(obs[0][0])
        return [self.fertilization_dic[dap] if dap in self.fertilization_dic else 0]

    def predict(self, obs, state=None, deterministic=None):
        action = self._policy(obs)
        action = normalize_action((self.env.action_low, self.env.action_high), action)

        return np.array([action], dtype=np.float32), obs


# evaluation and plotting functions
def evaluate(agent, n_episodes=10):
    # Create eval env
    eval_args = {
        'run_dssat_location': '/opt/dssat_pdi/run_dssat',
        'mode': 'fertilization',
        'seed': 456,
        'random_weather': True,
    }
    env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_args)))

    returns, _ = evaluate_policy(
        agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True)
    
    env.close()

    return returns

# evaluate agents
null_agent = NullAgent(env)
print('Evaluating Null agent...')
null_returns = evaluate(null_agent)
print('Done')

print('Evaluating PPO agent...')
ppo_returns = evaluate(ppo_agent)
print('Done')

expert_agent = ExpertAgent(env)
print('Evaluating Expert agent...')
expert_returns = evaluate(expert_agent)
print('Done')

# write results to a file to be loaded for display
data = [('null', null_returns), ('ppo', ppo_returns), ('expert', expert_returns)]
with open("results.pkl", "wb") as result_file:
    pickle.dump(data, result_file)

# Cleanup
env.close()
