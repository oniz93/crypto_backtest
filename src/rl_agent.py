"""
rl_agent.py
-----------
This module defines the reinforcement learning (RL) agent.
The agent uses a hybrid network architecture that processes the input as follows:
    1. A CNN layer reduces the high-dimensional input features.
    2. A TCN (Temporal Convolutional Network) block captures temporal patterns.
    3. A GRU layer further models sequential dependencies.
    4. A final fully connected layer produces Q-values for each action.
This architecture is designed to be much smaller (and faster) than using LSTM layers,
so that the model can process 1000 steps in less than 5 seconds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ---------------------------
# Hybrid Q-Network: CNN -> TCN -> GRU -> Fully Connected Layer
# ---------------------------
class HybridQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, cnn_channels=32,
                 tcn_channels=32, gru_hidden_size=32,
                 num_tcn_layers=1, num_gru_layers=1, seq_length=1440):
        """
        Hybrid Q-Network that implements a processing pipeline:
        CNN -> TCN -> GRU -> Fully Connected Layer.

        Parameters:
            state_dim (int): Number of input features per timestep (e.g. ~1000).
            action_dim (int): Number of possible actions.
            cnn_channels (int): Number of output channels for the CNN layer.
            tcn_channels (int): Number of output channels for the TCN block.
            gru_hidden_size (int): Hidden size for the GRU layer.
            num_tcn_layers (int): Number of dilated convolution layers in the TCN block.
            num_gru_layers (int): Number of GRU layers.
        """
        super(HybridQNetwork, self).__init__()
        self.state_dim = state_dim
        self.seq_length = seq_length
        self.cnn = nn.Conv1d(state_dim, cnn_channels, kernel_size=3, padding=1)
        tcn_layers = [nn.Conv1d(cnn_channels, tcn_channels, kernel_size=3, padding=2, dilation=2),
                      nn.ReLU()]
        self.tcn = nn.Sequential(*tcn_layers)
        self.gru = nn.GRU(tcn_channels, gru_hidden_size, num_layers=num_gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, action_dim)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the hybrid network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, state_dim).
            hidden_state: Not used here.

        Returns:
            tuple: (q_values, hidden_state) where q_values is of shape (batch, action_dim).
        """
        # Save the last timestep of the input state to access extra features.
        # The extra features appended by the environment are: 
        # [norm_adjusted_buy, norm_adjusted_sell, inventory, cash_ratio, norm_entry_diff]
        # We assume that the "inventory" (a proxy for whether entry_price is set) is at index: state_dim - 3.
        if x.dim() == 2:  # (batch_size, state_dim)
            x = x.unsqueeze(1).repeat(1, self.seq_length, 1)  # Expand to (batch_size, seq_length, state_dim)
        last_state = x[:, -1, :]  # (batch_size, state_dim)
        x = x.permute(0, 2, 1)  # (batch_size, state_dim, seq_length)
        x = torch.relu(self.cnn(x))  # (batch_size, cnn_channels, seq_length)
        x = self.tcn(x)  # (batch_size, tcn_channels, seq_length)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, tcn_channels)
        gru_out, _ = self.gru(x)  # (batch_size, seq_length, gru_hidden_size)
        q_values = self.fc(torch.relu(gru_out[:, -1, :]))  # (batch_size, action_dim)

        entry_price = last_state[:, self.state_dim - 1]  # (batch_size,)
        has_position = (torch.abs(entry_price) > 1e-6).float()
        q_values[:, 1] = q_values[:, 1] * (1.0 - has_position) + has_position * -1e6
        discourage_sell = (entry_price < 0.0).float() * has_position
        q_values[:, 2] = q_values[:, 2] * has_position + (1.0 - has_position) * -1e6 - discourage_sell * 0.1
        return q_values, None

# ---------------------------
# DQN Agent Using the HybridQNetwork
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, seq_length=1440, buffer_capacity=5000):
        """
        DQN Agent that uses the HybridQNetwork (CNN -> TCN -> GRU -> FC) for Q-learning.

        Parameters:
            state_dim (int): Number of features per timestep.
            action_dim (int): Number of possible actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay factor for exploration rate.
            epsilon_min (float): Minimum exploration rate.
            seq_length (int): Length of the input history (number of timesteps).
            buffer_capacity (int): Capacity of the replay buffer.
            per_alpha (float): Prioritization exponent for the replay buffer.
            per_beta (float): Initial importance-sampling exponent.
            per_beta_increment (float): Increment for per_beta after each update.
        """
        # Choose device.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seq_length = seq_length
        self.q_net = HybridQNetwork(state_dim, action_dim, seq_length=seq_length).to(self.device)
        self.target_net = HybridQNetwork(state_dim, action_dim, seq_length=seq_length).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None
        self.buffer = deque(maxlen=buffer_capacity)  # Efficient deque
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

    def _ensure_history(self, state):
        """
        Ensure that the input state has shape (seq_length, state_dim).

        If a 1D state is provided, tile it to form a history.
        If the state has a different number of timesteps, pad or truncate it.

        Parameters:
            state (np.array): Input state.

        Returns:
            np.array: State with shape (seq_length, state_dim).
        """
        state = np.asarray(state, dtype=np.float32)  # Faster conversion
        if state.ndim == 1:
            return np.tile(state, (self.seq_length, 1))
        pad_size = self.seq_length - state.shape[0]
        if pad_size > 0:
            return np.pad(state, ((0, pad_size), (0, 0)), mode='edge')
        return state[:self.seq_length, :]

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy, ensuring random actions respect position constraints.

        Parameters:
            state (np.array): Input state (either 1D or 2D).

        Returns:
            int: Chosen action index (0=hold, 1=buy, 2=sell).
        """
        state = self._ensure_history(state)  # Ensure state has shape (seq_length, state_dim)
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Shape: (1, seq_length, state_dim)

        # Extract entry_price from the last timestep of the state (index state_dim - 1)
        entry_price_index = self.state_dim - 1
        entry_price = state[-1, entry_price_index]  # Last timestep, entry_price feature

        # Define a small threshold for considering entry_price as zero (handles float precision)
        EPSILON = 1e-6
        has_position = abs(entry_price) > EPSILON  # True if a position is open

        # Define valid actions based on position status
        valid_actions = [0]  # Hold is always valid
        if not has_position:
            valid_actions.append(1)  # Buy is valid if no position
        if has_position:
            valid_actions.append(2)  # Sell is valid if position exists

        if np.random.rand() < self.epsilon:
            # Exploration: randomly select from valid actions
            action = np.random.choice(valid_actions)
        else:
            # Exploitation: use Q-network to select the best action
            with torch.no_grad():
                q_values, _ = self.q_net(state_tensor)
                # Mask invalid actions (though forward already does this, we reinforce it here)
                q_values = q_values.cpu().numpy()[0]  # Shape: (action_dim,)
                if has_position:
                    q_values[1] = float('-inf')  # Mask buy if position exists
                else:
                    q_values[2] = float('-inf')  # Mask sell if no position
                action = np.argmax(q_values)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Parameters:
            state, next_state (np.array): Input states (1D or 2D) that will be processed.
            action (int): Action taken.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        state = self._ensure_history(state)
        next_state = self._ensure_history(next_state)
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > 10000:
            self.buffer.pop(0)

    def update_policy(self):
        """
        Sample a batch of transitions from the replay buffer and update the Q-network.
        Uses mixed precision training if available and applies gradient clipping.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples yet

        # (If using PER, you would sample using that mechanism here.)
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)  # (batch, seq_length, state_dim)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        if self.scaler is not None:
            with torch.amp.autocast(device_type=self.device.type):
                q_values, _ = self.q_net(states)
                q_values = q_values.gather(1, actions)
                with torch.no_grad():
                    max_next_q, _ = self.target_net(next_states)
                    max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                    target = rewards + self.gamma * max_next_q * (1 - dones)
                loss = ((q_values - target) ** 2).mean()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(1, actions)
            with torch.no_grad():
                max_next_q, _ = self.target_net(next_states)
                max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                target = rewards + self.gamma * max_next_q * (1 - dones)
            loss = ((q_values - target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.scheduler.step(loss)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_policy_from_batch(self, batch):
        """
        Update the Q-network using a given mini-batch of transitions.

        Each transition is a tuple: (state, action, reward, next_state, done).
        This method ensures that each state is processed to have the required history
        (i.e. shape: (seq_length, state_dim)) and then performs a gradient descent step
        to minimize the TD error. Mixed precision training and gradient clipping are used
        if available.

        Parameters:
            batch (list): A list of transitions, each a tuple:
                          (state, action, reward, next_state, done)
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)

        self.optimizer.zero_grad()
        # Use autocast only if scaler is available, otherwise proceed normally
        if self.scaler:
            with torch.amp.autocast(device_type=self.device.type):
                q_values, _ = self.q_net(states)
                q_values = q_values.gather(1, actions)
                with torch.no_grad():
                    max_next_q, _ = self.target_net(next_states)
                    target = rewards + self.gamma * max_next_q.max(1)[0].unsqueeze(1) * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(1, actions)
            with torch.no_grad():
                max_next_q, _ = self.target_net(next_states)
                target = rewards + self.gamma * max_next_q.max(1)[0].unsqueeze(1) * (1 - dones)
            loss = nn.functional.mse_loss(q_values, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
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
