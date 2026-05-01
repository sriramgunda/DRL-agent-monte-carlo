import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.gamma = 0.9

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward + self.gamma * torch.max(self.model(next_state)).item()

            output = self.model(state)[action]
            loss = (output - target) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

    def get_q_values(self, state):
        state = torch.FloatTensor(state)
        q_values = self.model(state).detach().numpy()
        return q_values

    def save(self, path="data/dqn_model.pth"):
        os.makedirs("data", exist_ok=True)
        torch.save(self.model.state_dict(), path)


    def load(self, path="data/dqn_model.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print("DQN model loaded")
        else:
            print("No DQN model found, training from scratch")