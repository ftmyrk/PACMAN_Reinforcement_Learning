import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000
LR = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
TARGET_UPDATE_FREQ = 1000

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(6*7*64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_shape, num_actions).to(self.device)
        self.target_network = QNetwork(state_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                return self.q_network(state).max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        return random.sample(self.memory, BATCH_SIZE)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.sample_batch()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float32)

        q_values = self.q_network(state_batch).gather(1, action_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (GAMMA * next_q_values * (1 - done_batch))

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def preprocess_frame(frames):
    frame = frames[0] if isinstance(frames, tuple) else frames
    if len(frame.shape) == 3:
        frame = np.mean(frame, axis=2).astype(np.uint8)
    frame = frame[1:176:2, ::2]
    return np.expand_dims(frame, axis=0) / 255.0

def train(agent, env, num_episodes):
    episode_rewards = []
    for episode in range(num_episodes):
        state = preprocess_frame(env.reset())
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_frame(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return episode_rewards

if __name__ == "__main__":
    env = gym.make('ALE/MsPacman-v5', frameskip=4)
    num_actions = env.action_space.n
    state_shape = (1, 88, 80)  # preprocessed frame shape

    agent = DQNAgent(state_shape, num_actions)
    num_episodes = 1000
    rewards = train(agent, env, num_episodes)

    # Plotting
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
