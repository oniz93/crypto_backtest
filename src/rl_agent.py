"""
rl_agent.py
-----------
This module defines the reinforcement learning (RL) agent.
It implements a recurrent neural network using GRU layers (with an input projection layer)
to capture dependencies in the data. Several training optimizations are included,
such as gradient clipping, learning rate scheduling, mixed precision training, and
prioritized experience replay (PER).
This agent is used within the genetic algorithm to evaluate individuals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# SumTree and Prioritized Replay Buffer Implementation
# ---------------------------
class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    Used to store priorities and sample them in O(log n) time.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        """Add a new data point with its priority."""
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Update the tree with a new priority and propagate the change."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate the change through tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, s):
        """
        Find the leaf on the tree corresponding to the value s.
        Returns: (tree index, priority, data)
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = idx
                break
            else:
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer that uses a SumTree.
    """
    def __init__(self, capacity, alpha=0.6):
        """
        Parameters:
            capacity (int): Maximum number of transitions to store.
            alpha (float): How much prioritization is used (0 - no prioritization, 1 - full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)

    def add(self, error, sample):
        """
        Add a new sample with priority based on error.
        The priority is computed as (abs(error) + epsilon) ** alpha.
        """
        epsilon = 1e-6
        priority = (abs(error) + epsilon) ** self.alpha
        self.tree.add(priority, sample)

    def sample(self, n, beta=0.4):
        """
        Sample a batch of n transitions.
        beta: importance-sampling (IS) exponent (0 - no corrections, 1 - full correction).
        Returns:
            batch: list of transitions
            idxs: list of tree indices for updating priorities later
            is_weights: array of importance-sampling weights for the batch
        """
        batch = []
        idxs = []
        segment = self.tree.total_priority / n
        is_weights = np.empty((n, 1), dtype=np.float32)

        # To normalize IS weights, get the minimum probability
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # Fix: Avoid division by zero
        if p_min == 0:
            max_weight = 1.0
        else:
            max_weight = (p_min * n) ** (-beta)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            sampling_prob = priority / self.tree.total_priority
            is_weight = (sampling_prob * n) ** (-beta)
            is_weight /= max_weight
            batch.append(data)
            idxs.append(idx)
            is_weights[i, 0] = is_weight

        return batch, idxs, is_weights

    def update(self, idx, error):
        """
        Update the priority of a transition.
        """
        epsilon = 1e-6
        priority = (abs(error) + epsilon) ** self.alpha
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries

# ---------------------------
# GRU-based Q-Network with Input Projection
# ---------------------------
class GRUQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 reduced_dim=256, hidden_size_gru=256, gru_layers=1):
        """
        Q-Network with an input projection layer followed by GRU layers.

        Parameters:
            state_dim (int): Number of features per timestep (e.g. 800-1000).
            action_dim (int): Number of possible actions.
            reduced_dim (int): Dimensionality after projection.
            hidden_size_gru (int): Hidden size for the GRU layer.
            gru_layers (int): Number of stacked GRU layers.
        """
        super(GRUQNetwork, self).__init__()
        # Project high-dimensional input into a lower-dimensional space.
        self.input_projection = nn.Linear(state_dim, reduced_dim)
        # GRU to process the projected input.
        self.gru = nn.GRU(input_size=reduced_dim,
                          hidden_size=hidden_size_gru,
                          num_layers=gru_layers,
                          batch_first=True)
        # Final fully connected layer to output Q-values.
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
        # First, reduce dimensionality.
        x = torch.relu(self.input_projection(x))
        # Pass through GRU layers.
        gru_out, _ = self.gru(x)  # Shape: (batch, seq_len, hidden_size_gru)
        # Select the output of the last timestep.
        last_out = gru_out[:, -1, :]  # Shape: (batch, hidden_size_gru)
        # Apply ReLU and map to Q-values.
        q_values = self.fc(torch.relu(last_out))
        return q_values, None

# ---------------------------
# DQN Agent with Optimizations and Prioritized Replay Buffer
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.90,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, seq_length=1440,
                 buffer_capacity=10000, per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001):
        """
        DQN Agent using the GRU Q-Network with optimizations and PER.

        Parameters:
            state_dim (int): Number of features per timestep.
            action_dim (int): Number of actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay factor for exploration.
            epsilon_min (float): Minimum exploration rate.
            seq_length (int): History length (number of timesteps).
            buffer_capacity (int): Capacity of the replay buffer.
            per_alpha (float): Prioritization exponent.
            per_beta (float): Initial importance-sampling exponent.
            per_beta_increment (float): Increment for beta per update.
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

        # Initialize Q-network and target network.
        self.q_net = GRUQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = GRUQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        # Set up the optimizer.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # Add a learning rate scheduler.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=10)
        # Use torch.amp for mixed precision training (if using CUDA).
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        # Initialize the prioritized replay buffer.
        self.buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

        # PER parameters.
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment

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
            state = np.tile(state, (self.seq_length, 1))
        elif state.shape[0] != self.seq_length:
            pad_size = self.seq_length - state.shape[0]
            if pad_size > 0:
                pad = np.repeat(state[0:1, :], pad_size, axis=0)
                state = np.concatenate([state, pad], axis=0)
            else:
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
        state = self._ensure_history(state)  # (seq_length, state_dim)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # (1, seq_length, state_dim)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values, _ = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the prioritized replay buffer.
        The initial priority is set to the maximum priority in the buffer (or 1 if the buffer is empty).
        """
        state = self._ensure_history(state)
        next_state = self._ensure_history(next_state)
        transition = (state, action, reward, next_state, done)
        # Use max priority so that new transitions are likely to be sampled.
        if len(self.buffer) > 0:
            max_priority = np.max(self.buffer.tree.tree[-self.buffer.capacity:])
            if max_priority == 0:
                max_priority = 1.0
        else:
            max_priority = 1.0
        self.buffer.add(max_priority, transition)

    def update_policy(self):
        """
        Sample a batch from the replay buffer using PER, update the network, and update priorities.
        Applies gradient clipping and uses mixed precision training if available.
        """
        if len(self.buffer) < self.batch_size:
            return  # Not enough samples yet

        # Increase beta (for importance-sampling weights) gradually.
        self.per_beta = np.min([1.0, self.per_beta + self.per_beta_increment])

        batch, idxs, is_weights = self.buffer.sample(self.batch_size, beta=self.per_beta)

        # Unpack batch transitions.
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)  # (batch, seq_length, state_dim)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        self.optimizer.zero_grad()

        # Use mixed precision if available.
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                q_values, _ = self.q_net(states)
                q_values = q_values.gather(1, actions)
                with torch.no_grad():
                    max_next_q, _ = self.target_net(next_states)
                    max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                    target = rewards + self.gamma * max_next_q * (1 - dones)
                td_errors = q_values - target
                loss = (is_weights * (td_errors ** 2)).mean()
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(1, actions)
            with torch.no_grad():
                max_next_q, _ = self.target_net(next_states)
                max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                target = rewards + self.gamma * max_next_q * (1 - dones)
            td_errors = q_values - target
            loss = (is_weights * (td_errors ** 2)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.step_count += 1
        # Update the target network periodically.
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        # Update the learning rate scheduler with the current loss.
        self.scheduler.step(loss)
        # Decay the exploration rate.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update priorities in the replay buffer.
        # Use the absolute TD error as the new priority.
        td_errors_np = td_errors.detach().cpu().numpy().squeeze()
        for idx, error in zip(idxs, td_errors_np):
            self.buffer.update(idx, error)

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
