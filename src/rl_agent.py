# src/rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, gru_layers=1):
        """
        A Q-network that uses a GRU layer to capture temporal dependencies.
        - state_dim: number of input features.
        - action_dim: number of actions.
        - hidden_size: number of hidden units in the GRU.
        - gru_layers: number of GRU layers.
        """
        super(QNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        # The GRU expects input of shape (batch, seq_len, state_dim).
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden_size,
                          num_layers=gru_layers, batch_first=True)
        # Final fully connected layer to produce Q-values.
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, x, hidden_state=None):
        """
        Forward pass:
          - x: tensor of shape (batch, seq_len, state_dim).
          - hidden_state: initial hidden state for the GRU.
        Returns:
          - q_values: tensor of shape (batch, action_dim).
          - hidden_state: final hidden state from the GRU.
        """
        if hidden_state is None:
            hidden_state = torch.zeros(self.gru_layers, x.size(0), self.hidden_size).to(x.device)
        gru_out, hidden_state = self.gru(x, hidden_state)
        # Use the output from the last time step.
        last_out = gru_out[:, -1, :]
        q_values = self.fc(torch.relu(last_out))
        return q_values, hidden_state


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.90,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seq_length=1440):
        """
        DQN Agent that uses a GRU-based Q-network.
        - seq_length: the length of the state sequence input.
          (If you plan to use a history of states, set seq_length > 1.
           For a single state, seq_length=1.)
        """
        # Set device (change "cpu" to "cuda" if you have a GPU) or "mps" for apple-silicon
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "mps"
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seq_length = seq_length  # sequence length for GRU input

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = []
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        The state (a 1D array) is reshaped into a sequence with length 1.
        """
        state = torch.FloatTensor(state).to(self.device)
        # Reshape to (batch, seq_len, state_dim). For a single state, seq_len = 1.
        state = state.unsqueeze(0).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values, _ = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def update_policy(self):
        if len(self.buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to tensors and add sequence dimension.
        states = torch.from_numpy(np.stack(states)).float().to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values, _ = self.q_net(states)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
            max_next_q, _ = self.target_net(next_states)
            max_next_q = max_next_q.max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = ((q_values - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"RL Agent's weights saved to {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"RL Agent's weights loaded from {path}")
