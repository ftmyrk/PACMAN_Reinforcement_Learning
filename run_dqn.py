from dqn import DQN
from buffer import ReplayBuffer

import ale_py

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import os

# Ensure the Plots directory exists
os.makedirs('./Plots', exist_ok=True)


if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('Found device at: {}'.format(device))

# gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")
config = {
    'lr': 0.0005,
    'C': 60,
    'discount': 0.99,
    'batch_size': 32,
    'replay_buffer_size': 100000,
    'eps_min': 0.01,
    'eps_max': 1.0,
    'eps_len': 4000,
    'seed': 1,
    'state_shape': env.observation_space.shape,
    'action_size': env.action_space.n,
    'n_episodes': 1000,
    'gamma': 0.99,
    'epsilon_decay': 0.995,
    'target_update_freq': 10,
    'dims_hidden_neurons': (64, 64),
}

dqn = DQN(config)
buffer = ReplayBuffer(config)


episode_rewards = []
steps_list = []
epsilon_list = []

steps = 0  # total number of steps
for i_episode in range(20):
    state = env.reset()[0]
    state = np.reshape(state, [1] + list(config['state_shape']))
    done = False
    truncated = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return
    while done is False and truncated is False:
        # env.render()  # render to screen, not working for jupyter

        obs = torch.tensor(state)  # observe the environment state

        action = dqn.act_probabilistic(state[None, :])  # take action

        next_obs, reward, done, info,_ = env.step(action)  # environment advance to next step

        buffer.append_memory(obs=obs,  # put the transition to memory
                             action=torch.from_numpy(np.array([action])),
                             reward=torch.from_numpy(np.array([reward])),
                             next_obs=torch.from_numpy(next_obs),
                             done=done)

        dqn.update(buffer)  # agent learn

        t += 1
        steps += 1
        ret += reward  # update episodic return
        if done or truncated:
            print(f"Episode {i_episode} finished after {t+1} timesteps with return {ret}")
            episode_rewards.append(ret)
            steps_list.append(steps)
            epsilon_list.append(dqn.epsilon)
            break

env.close()

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('./Plots/episode_rewards.png')
plt.close()

plt.subplot(3, 1, 2)
plt.plot(steps_list, episode_rewards)
plt.title('Rewards vs Steps')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.savefig('./Plots/rewards_vs_steps.png')
plt.close()

plt.subplot(3, 1, 3)
plt.plot(epsilon_list)
plt.title('Epsilon Decay')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.savefig('./Plots/epsilon_decay.png')
plt.close()