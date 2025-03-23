"""
rl_agent.py
-----------
This module defines the reinforcement learning (RL) agent.
... (rest of the header) ...
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
import sys
import os

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# Set up logging
# logger = logging.getLogger('GeneticOptimizer')
# logger.setLevel(logging.DEBUG)
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
# pipe_path = '/tmp/genetic_optimizer_logpipe'
# if not os.path.exists(pipe_path):
#     os.mkfifo(pipe_path)
# try:
#     pipe = open(pipe_path, 'w')
#     pipe_handler = logging.StreamHandler(pipe)
#     pipe_handler.setLevel(logging.DEBUG)
#     pipe_handler.setFormatter(formatter)
#     logger.addHandler(pipe_handler)
# except Exception as e:
#     logger.error(f"Failed to open named pipe {pipe_path}: {e}")
#

# ---------------------------
# Hybrid Q-Network: CNN -> TCN -> GRU -> Fully Connected Layer
# ---------------------------
class HybridQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, cnn_channels=16,
                 tcn_channels=16, gru_hidden_size=16, num_tcn_layers=1,
                 num_gru_layers=1, seq_length=720):
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
        # CNN layer
        self.cnn = nn.Conv1d(state_dim, cnn_channels, kernel_size=3, padding=1)
        # Multi-layer TCN
        tcn_layers = []
        for i in range(num_tcn_layers):
            dilation = 2 ** i  # Dilations: 1, 2, 4
            padding = (3 - 1) * dilation // 2
            in_channels = cnn_channels if i == 0 else tcn_channels
            tcn_layers.append(nn.Conv1d(in_channels, tcn_channels, kernel_size=3,
                                      padding=padding, dilation=dilation))
            tcn_layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*tcn_layers)
        # GRU layer
        self.gru = nn.GRU(tcn_channels, gru_hidden_size, num_layers=num_gru_layers,
                         batch_first=True)
        # Output layer
        self.fc = nn.Linear(gru_hidden_size, action_dim)
        # Forward pass remains similar but operates on updated dimensions

    def forward(self, x, hidden_state=None, env=None):
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
        x = x.float()

        if x.dim() == 2:
            x = x.unsqueeze(0)
        last_state = x[:, -1, :]  # (batch_size, state_dim)
        x = x.transpose(1, 2)  # (batch, state_dim, seq_length) for CNN

        # Apply CNN layer
        x = self.cnn(x)  # (batch, cnn_channels, seq_length)
        x = self.tcn(x)  # (batch, tcn_channels, seq_length)
        x = x.transpose(1, 2)  # (batch, seq_length, tcn_channels) for GRU

        if hidden_state is None:
            hidden_state = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)

        # Pass through GRU
        x, hidden_state = self.gru(x, hidden_state)
        x = x[:, -1, :]  # Take last timestep (batch, gru_hidden_size)

        # Compute Q-values
        q_values = self.fc(x)  # (batch, action_dim)

        # Adjust Q-values based on gain_loss and inventory if env is provided
        if env is not None:
            gain_loss = torch.tensor([float(env.gain_loss)], dtype=torch.float32, device=x.device).expand(x.size(0))  # (batch,)
            inventory = torch.tensor([float(env.inventory)], dtype=torch.float32, device=x.device).expand(x.size(0))  # (batch,)
            has_position = (torch.abs(inventory) > 1e-6).float()  # (batch,)

            # Discourage invalid actions
            q_values[:, 1] = q_values[:, 1] * (1.0 - has_position) - has_position * 1e6  # Buy when position exists
            q_values[:, 2] = q_values[:, 2] * has_position - (1.0 - has_position) * 1e6  # Sell when no position

            # Inhibit sell (action 2) when gain_loss is negative
            discourage_sell = (gain_loss < 0.0).float() * has_position
            fee_penalty = torch.tensor([float(env.total_fees) * 1.0], dtype=torch.float32, device=x.device).expand(
                x.size(0))  # Scale penalty by fees
            q_values[:, 2] -= discourage_sell * (2.0 * torch.abs(gain_loss) + fee_penalty)  # Reduce sell Q-value proportionally

        return q_values, hidden_state

# ---------------------------
# DQN Agent Using the HybridQNetwork
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 seq_length=120, buffer_capacity=100000, use_cudf=False):
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
            seq_length (int): Length of the inpout history (number of timesteps).
            buffer_capacity (int): Capacity of the replay buffer.
            per_alpha (float): Prioritization exponent for the replay buffer.
            per_beta (float): Initial importance-sampling exponent.
            per_beta_increment (float): Increment for per_beta after each update.
        """
        # Choose device.
        if torch.cuda.is_available():
            device = "cuda"
        # elif torch.backends.mps.is_available():
        #     device = "mps"
        else:
            device = "cpu"

        self.use_cudf=use_cudf
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seq_length = seq_length
        self.buffer = deque(maxlen=buffer_capacity)
        self.update_target_every = 1000
        # Initialize q_net and target_net with updated HybridQNetwork
        self.q_net = HybridQNetwork(state_dim, action_dim)
        self.target_net = HybridQNetwork(state_dim, action_dim)

        self.q_net.to(self.device)  # Add this line
        self.target_net.to(self.device)  # Add this line

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.env = None
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
        if self.use_cudf and HAS_CUPY and isinstance(state, cp.ndarray):
            state_cupy = state.astype(cp.float32) # keep as cupy array
            if state_cupy.ndim == 1:
                return cp.tile(state_cupy, (self.seq_length, 1))
            pad_size = self.seq_length - state_cupy.shape[0]
            if pad_size > 0:
                return cp.pad(state_cupy, ((0, pad_size), (0, 0)), mode='edge')
            return state_cupy[:self.seq_length, :]
        else: # numpy path
            state_numpy = np.asarray(state, dtype=np.float32)  # Faster conversion
            if state_numpy.ndim == 1:
                return np.tile(state_numpy, (self.seq_length, 1))
            pad_size = self.seq_length - state_numpy.shape[0]
            if pad_size > 0:
                return np.pad(state_numpy, ((0, pad_size), (0, 0)), mode='edge')
            return state_numpy[:self.seq_length, :]

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy, ensuring random actions respect position constraints.

        Parameters:
            state (np.array): Input state (either 1D or 2D).

        Returns:
            int: Chosen action index (0=hold, 1=buy, 2=sell).
        """
        state = self._ensure_history(state)
        if self.use_cudf and HAS_CUPY and isinstance(state, cp.ndarray):
            state_tensor = torch.from_numpy(state.get()).float().to(self.device).unsqueeze(0) # Convert cupy array to numpy then torch tensor
            inventory = state[-1, self.state_dim - 2].get()  # Inventory is now at index -2, get value from cupy array
        else:
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            inventory = state[-1, self.state_dim - 2]  # Inventory is now at index -2
        has_position = abs(inventory) > 1e-6

        valid_actions = [0]
        if not has_position:
            valid_actions.append(1)
        if has_position:
            valid_actions.append(2)

        if np.random.rand() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values, _ = self.q_net(state_tensor, env=self.env)  # Pass env to access gain_loss
                q_values = q_values.cpu().numpy()[0]
                if has_position:
                    q_values[1] = float('-inf')
                else:
                    q_values[2] = float('-inf')
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

    def update_policy_from_batch(self, batch):
        """
        Update the Q-network using a given mini-batch of transitions.

        Parameters:
            batch (list): A list of transitions, each a tuple:
                          (state, action, reward, next_state, done)
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        processed_states = [self._ensure_history(s) for s in states]
        processed_next_states = [self._ensure_history(s) for s in next_states]

        # Convert CuPy to NumPy if necessary
        processed_states_np = [cp.asnumpy(s) if isinstance(s, cp.ndarray) else s for s in processed_states]
        processed_next_states_np = [cp.asnumpy(s) if isinstance(s, cp.ndarray) else s for s in processed_next_states]

        # Stack and convert to tensors
        states = torch.from_numpy(np.stack(processed_states_np)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(processed_next_states_np)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)

        self.optimizer.zero_grad()

        # Compute Q-values for current states
        q_values, _ = self.q_net(states, hidden_state=None, env=self.env)
        q_values = q_values.gather(1, actions)

        with torch.no_grad():
            # Compute Q-values for next states using q_net
            q_values_next, _ = self.q_net(next_states, hidden_state=None, env=self.env)
            next_actions = q_values_next.argmax(dim=1, keepdim=True)  # Shape: [32, 1], values in {0, 1, 2}

            # Compute target Q-values
            target_q_values, _ = self.target_net(next_states, hidden_state=None, env=self.env)
            max_next_q = target_q_values.gather(1, next_actions)
            target = rewards + self.gamma * max_next_q * (1 - dones)

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