# src/rl_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CombinedQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_size_lstm=512, hidden_size_gru=512,
                 lstm_layers=2, gru_layers=2):
        """
        A Q-network that combines an LSTM layer followed by a GRU layer.
        Both layers process a history of timesteps (e.g. 1440 timestamps).

        Args:
            state_dim (int): Number of features per timestep (e.g. ~1000).
            action_dim (int): Number of actions.
            hidden_size_lstm (int): Hidden size of the LSTM layer.
            hidden_size_gru (int): Hidden size of the GRU layer.
            lstm_layers (int): Number of LSTM layers.
            gru_layers (int): Number of GRU layers.
        """
        super(CombinedQNetwork, self).__init__()
        # LSTM block to capture long-term dependencies.
        self.lstm = nn.LSTM(input_size=state_dim,
                            hidden_size=hidden_size_lstm,
                            num_layers=lstm_layers,
                            batch_first=True)
        # GRU block to refine the output.
        self.gru = nn.GRU(input_size=hidden_size_lstm,
                          hidden_size=hidden_size_gru,
                          num_layers=gru_layers,
                          batch_first=True)
        # Fully connected layer to produce Q-values.
        self.fc = nn.Linear(hidden_size_gru, action_dim)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, state_dim). Here, seq_len should be 1440.
            hidden_state: Not used in this example.
        Returns:
            q_values: Tensor of shape (batch, action_dim).
        """
        # Pass through LSTM.
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size_lstm)
        # Pass LSTM outputs through GRU.
        gru_out, _ = self.gru(lstm_out)  # (batch, seq_len, hidden_size_gru)
        # Use the output at the final timestep.
        last_out = gru_out[:, -1, :]  # (batch, hidden_size_gru)
        # Apply a ReLU activation and then the final layer.
        q_values = self.fc(torch.relu(last_out))
        return q_values, None


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.90,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seq_length=1440):
        """
        DQN Agent that uses a combined LSTM+GRU network.

        Args:
            state_dim (int): Feature dimension (e.g. ~1000).
            action_dim (int): Number of actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Multiplicative decay for epsilon.
            epsilon_min (float): Minimum epsilon.
            seq_length (int): The length of the input history (e.g. 1440 timesteps).
        """
        # Use CUDA if available.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seq_length = seq_length

        # Create the combined recurrent network.
        self.q_net = CombinedQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = CombinedQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = []
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Expects state to be a 2D array of shape (seq_length, state_dim), e.g. (1440, 1000).
        A batch dimension is added.
        """
        state = torch.FloatTensor(state).to(self.device)
        # Now state is (seq_length, state_dim); add a batch dimension:
        state = state.unsqueeze(0)  # (1, seq_length, state_dim)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values, _ = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition. It is assumed that both state and next_state are 2D arrays
        of shape (seq_length, state_dim).
        """
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def update_policy(self):
        if len(self.buffer) < self.batch_size:
            return
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        # Instead of unsqueezing, we expect states to already be (seq_length, state_dim).
        # We need to stack them to shape (batch, seq_length, state_dim):
        states = torch.from_numpy(np.stack(states)).float().to(self.device)  # (batch, seq_length, state_dim)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
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
