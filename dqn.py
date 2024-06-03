import torch
import torch.nn as nn
from typing import Tuple
from numpy.random import binomial, choice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN:
    def __init__(self, config):
        torch.manual_seed(config['seed'])
        
        self.lr = config['lr']
        self.C = config['C']
        self.eps_len = config['eps_len']
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']
        self.batch_size = config['batch_size']

        self.state_shape = config['state_shape']
        self.action_size = config['action_size']

        self.Q = QNetwork(self.state_shape, self.action_size).to(device)
        self.Q_tar = QNetwork(self.state_shape, self.action_size).to(device)

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

    def act_probabilistic(self, observation):
        observation = observation.to(device)
        # epsilon greedy:
        first_term = self.eps_max * (self.eps_len - self.training_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.action_size)
        else:
            self.Q.eval()
            Q = self.Q(observation)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation):
        observation = observation.to(device)
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=1)
        self.Q.train()
        return a.item()

    def update(self, buffer):
        t = buffer.sample(self.batch_size)

        s = t.obs.to(device)
        a = t.action.to(device)
        r = t.reward.squeeze(-1).to(device)
        sp = t.next_obs.to(device)
        done = t.done.float().squeeze(-1).to(device)

        self.training_step += 1

        q_values = self.Q(s)
        q_next = self.Q_tar(sp)

        q_next_max = q_next.max(dim=1)[0]
        q_target = r + self.discount * q_next_max * (1 - done)

        if a.dim() == 1:
            a = a.unsqueeze(1)
            
        q_value = q_values.gather(1, a).squeeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_value, q_target.detach())

        self.optimizer_Q.zero_grad()
        loss.backward()
        self.optimizer_Q.step()

        if self.training_step % self.C == 0:
            self.update_target_model()

    def update_target_model(self):
        self.Q_tar.load_state_dict(self.Q.state_dict())

class QNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(256 * 88, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float32).squeeze(dim=1).permute(0, 3, 1, 2).to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)
