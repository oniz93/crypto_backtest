"""
rl_agent.py
-----------
This module defines the reinforcement learning (RL) agent.
It implements a recurrent neural network using GRU layers to capture dependencies in the data.
This agent is used within the genetic algorithm to evaluate individuals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GRUQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_size_gru=256, gru_layers=1):
        """
        Q-Network with GRU layers.

        Parameters:
            state_dim (int): Number of features per timestep (e.g. ~1000).
            action_dim (int): Number of possible actions.
            hidden_size_gru (int): Hidden size for the GRU layer.
            gru_layers (int): Number of stacked GRU layers.
        """
        super(GRUQNetwork, self).__init__()
        # GRU to process the input sequence.
        self.gru = nn.GRU(input_size=state_dim,
                          hidden_size=hidden_size_gru,
                          num_layers=gru_layers,
                          batch_first=True)
        # Final fully connected layer to output Q-values for each action.
        self.fc = nn.Linear(hidden_size_gru, action_dim)

    def forward(self, x, hidden_state=None):
        """
        Forward pass of the network.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, state_dim).
            hidden_state: Not used in this implementation.

        Returns:
            tuple: (q_values, hidden_state) where q_values is of shape (batch, action_dim).
        """
        # Pass the input through the GRU layers.
        gru_out, _ = self.gru(x)  # Shape: (batch, seq_len, hidden_size_gru)
        # Select the output of the last timestep.
        last_out = gru_out[:, -1, :]  # Shape: (batch, hidden_size_gru)
        # Apply ReLU activation and map to Q-values.
        q_values = self.fc(torch.relu(last_out))
        return q_values, None

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.90,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seq_length=1440):
        """
        DQN Agent using the GRU Q-Network.

        Parameters:
            state_dim (int): Number of features per timestep.
            action_dim (int): Number of actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay factor for exploration.
            epsilon_min (float): Minimum exploration rate.
            seq_length (int): History length (number of timesteps).
        """
        # Choose the appropriate device (CUDA if available, else CPU).
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seq_length = seq_length

        # Initialize the Q-network and target network.
        self.q_net = GRUQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = GRUQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        # Set up the optimizer.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # Initialize the replay buffer.
        self.buffer = []
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

    def _ensure_history(self, state):
        """
        Ensure that the input state has shape (seq_length, state_dim).

        If a 1D state (state_dim,) is provided, it is tiled to create a history.
        If the state has a different number of timesteps, it is padded or truncated.

        Parameters:
            state (np.array): Input state.

        Returns:
            np.array: State with shape (seq_length, state_dim).
        """
        state = np.array(state)
        if state.ndim == 1:
            # If state is 1D, replicate it to form a history.
            state = np.tile(state, (self.seq_length, 1))
        elif state.shape[0] != self.seq_length:
            pad_size = self.seq_length - state.shape[0]
            if pad_size > 0:
                # If state is too short, pad it by repeating the first row.
                pad = np.repeat(state[0:1, :], pad_size, axis=0)
                state = np.concatenate([state, pad], axis=0)
            else:
                # If state is too long, truncate it.
                state = state[:self.seq_length, :]
        return state

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Parameters:
            state (np.array): State input (1D or 2D).

        Returns:
            int: Selected action index.
        """
        # Ensure the state has the required history dimension.
        state = self._ensure_history(state)  # (seq_length, state_dim)
        # Add a batch dimension.
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # (1, seq_length, state_dim)
        if np.random.rand() < self.epsilon:
            # With probability epsilon, select a random action.
            return np.random.randint(self.action_dim)
        # Otherwise, select the action with the highest Q-value.
        with torch.no_grad():
            q_values, _ = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Parameters:
            state, next_state (np.array): States (1D or 2D) that will be processed.
            action (int): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        # Ensure both states have the required history.
        state = self._ensure_history(state)
        next_state = self._ensure_history(next_state)
        self.buffer.append((state, action, reward, next_state, done))
        # Limit the buffer size.
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def update_policy(self):
        """
        Sample a batch from the replay buffer and perform a gradient descent step to update the network.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples yet
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        # Stack the states into a single tensor.
        states = torch.from_numpy(np.stack(states)).float().to(self.device)  # (batch, seq_length, state_dim)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Get current Q-values.
        q_values, _ = self.q_net(states)
        q_values = q_values.gather(1, actions)
        # Compute target Q-values using the target network.
        with torch.no_grad():
            max_next_q, _ = self.target_net(next_states)
            max_next_q = max_next_q.max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * max_next_q * (1 - dones)
        # Compute loss as mean squared error.
        loss = ((q_values - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        # Periodically update the target network.
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        # Decay the exploration rate.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        """
        Save the Q-network weights to a file.

        Parameters:
            path (str): File path to save the model.
        """
        torch.save(self.q_net.state_dict(), path)
        print(f"RL Agent's weights saved to {path}")

    def load(self, path):
        """
        Load Q-network weights from a file.

        Parameters:
            path (str): File path from which to load the model.
        """
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"RL Agent's weights loaded from {path}")
