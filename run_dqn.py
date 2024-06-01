from dqn import DQN
from buffer import ReplayBuffer

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# Ensure the Plots directory exists
os.makedirs('./Plots', exist_ok=True)

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('Found device at: {}'.format(device))

env = gym.make('ALE/MsPacman-v5', render_mode="human")

config = {
    'dim_obs': env.observation_space.shape[0],  # Adjust to match environment observation space
    'dim_action': env.action_space.n,  # Adjust to match environment action space
    'dims_hidden_neurons': (128, 128),  # Adjusted Q network hidden layers
    'lr': 0.0005,  # learning rate
    'C': 100,  # copy steps
    'discount': 0.99,  # discount factor
    'batch_size': 32,
    'replay_buffer_size': 50000,
    'eps_min': 0.01,
    'eps_max': 1.0,
    'eps_len': 10000,
    'seed': 42,
    'device': device,
}

dqn = DQN(config)
buffer = ReplayBuffer(config)

# Variables for plotting
episode_rewards = []
steps_list = []
epsilon_list = []

steps = 0  # total number of steps
for i_episode in range(500):
    observation = env.reset()
    done = False
    truncated = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    while done is False and truncated is False:
        obs = torch.tensor(observation, dtype=torch.float32).to(device)  # observe the environment state

        action = dqn.act_probabilistic(obs[None, :])  # take action

        next_obs, reward, done, truncated, _ = env.step(action)  # environment advance to next step

        buffer.append_memory(obs=obs,  # put the transition to memory
                             action=torch.tensor([action], dtype=torch.int64).to(device),
                             reward=torch.tensor([reward], dtype=torch.float32).to(device),
                             next_obs=torch.tensor(next_obs, dtype=torch.float32).to(device),
                             done=done)

        dqn.update(buffer)  # agent learn

        t += 1
        steps += 1
        ret += reward  # update episodic return
        observation = next_obs

        if done or truncated:
            print(f"Episode {i_episode} finished after {t+1} timesteps with return {ret}")
            episode_rewards.append(ret)
            steps_list.append(steps)
            epsilon_list.append(dqn.epsilon)
            break

env.close()

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('./Plots/episode_rewards.png')

plt.subplot(3, 1, 2)
plt.plot(steps_list, episode_rewards)
plt.title('Rewards vs Steps')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.savefig('./Plots/rewards_vs_steps.png')

plt.subplot(3, 1, 3)
plt.plot(epsilon_list)
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.savefig('./Plots/epsilon_decay.png')

plt.tight_layout()
plt.savefig('./Plots/training_plots.png')  # Save all plots combined
