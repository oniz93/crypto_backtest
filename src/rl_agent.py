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

# ---------------------------
# Hybrid Q-Network: CNN -> TCN -> GRU -> Fully Connected Layer
# ---------------------------
class HybridQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 cnn_channels=64, tcn_channels=64, gru_hidden_size=64,
                 num_tcn_layers=2, num_gru_layers=1):
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
        # The input state is expected in shape: (batch, seq_length, state_dim).
        # We first permute it to (batch, state_dim, seq_length) so that we can apply 1D convolutions along the time axis.

        # ---------------------------
        # CNN Block: Reduce feature dimensionality
        # ---------------------------
        self.cnn = nn.Conv1d(in_channels=state_dim,
                             out_channels=cnn_channels,
                             kernel_size=3,
                             padding=1)  # padding=1 preserves sequence length

        # ---------------------------
        # TCN Block: Multiple dilated convolutions to capture temporal patterns
        # ---------------------------
        tcn_layers = []
        in_channels = cnn_channels
        # Create a series of convolutional layers with increasing dilation.
        for i in range(num_tcn_layers):
            dilation = 2 ** i  # Exponential dilation to enlarge the receptive field
            tcn_layers.append(
                nn.Conv1d(in_channels, tcn_channels,
                          kernel_size=3,
                          padding=dilation,
                          dilation=dilation)
            )
            tcn_layers.append(nn.ReLU())
            in_channels = tcn_channels
        self.tcn = nn.Sequential(*tcn_layers)

        # ---------------------------
        # GRU Block: Process the sequential output from TCN
        # ---------------------------
        # We first need to permute the TCN output back to (batch, seq_length, channels)
        self.gru = nn.GRU(input_size=tcn_channels,
                          hidden_size=gru_hidden_size,
                          num_layers=num_gru_layers,
                          batch_first=True)

        # ---------------------------
        # Fully Connected Layer: Map GRU output to Q-values for each action
        # ---------------------------
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
        # Permute input to shape (batch, state_dim, seq_length) for CNN.
        x = x.permute(0, 2, 1)
        # Apply CNN and a ReLU activation.
        x = torch.relu(self.cnn(x))  # (batch, cnn_channels, seq_length)
        # Apply TCN block.
        x = self.tcn(x)  # (batch, tcn_channels, seq_length)
        # Permute back to (batch, seq_length, tcn_channels) for GRU.
        x = x.permute(0, 2, 1)
        # Pass through the GRU layer(s); we only need the output.
        gru_out, _ = self.gru(x)  # (batch, seq_length, gru_hidden_size)
        # Select the output from the last timestep.
        last_out = gru_out[:, -1, :]  # (batch, gru_hidden_size)
        # Optionally apply ReLU and map to Q-values.
        q_values = self.fc(torch.relu(last_out))
        return q_values, None

# ---------------------------
# DQN Agent Using the HybridQNetwork
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.90,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 seq_length=1440, buffer_capacity=10000,
                 per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001):
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

        # Initialize the hybrid Q-network and the target network.
        self.q_net = HybridQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = HybridQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Set up the optimizer.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # Learning rate scheduler (optional).
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=10)
        # Mixed precision training support (if using CUDA).
        self.scaler = torch.amp.GradScaler() if self.device.type == "cuda" else None

        # (For simplicity, the PrioritizedReplayBuffer class is assumed to be defined elsewhere in your project.)
        # Here you would import it if needed:
        # from your_module import PrioritizedReplayBuffer
        # For now, we assume a simple list as a buffer.
        self.buffer = []  # Replace with PER buffer if available
        self.batch_size = 64
        self.update_target_every = 100
        self.step_count = 0

        # PER parameters.
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment

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
        state = np.array(state)
        if state.ndim == 1:
            # Tile the single state to form a sequence.
            state = np.tile(state, (self.seq_length, 1))
        elif state.shape[0] != self.seq_length:
            pad_size = self.seq_length - state.shape[0]
            if pad_size > 0:
                # Pad by repeating the first row.
                pad = np.repeat(state[0:1, :], pad_size, axis=0)
                state = np.concatenate([state, pad], axis=0)
            else:
                # Truncate the state to the required sequence length.
                state = state[:self.seq_length, :]
        return state

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy strategy.

        Parameters:
            state (np.array): Input state (either 1D or 2D).

        Returns:
            int: Chosen action index.
        """
        state = self._ensure_history(state)  # Ensure state has shape (seq_length, state_dim)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Add batch dimension: (1, seq_length, state_dim)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values, _ = self.q_net(state)
        return q_values.argmax().item()

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
            with torch.amp.autocast(device_type=self.device):
                q_values, _ = self.q_net(states)
                q_values = q_values.gather(1, actions)
                with torch.no_grad():
                    max_next_q, _ = self.target_net(next_states)
                    max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                    target = rewards + self.gamma * max_next_q * (1 - dones)
                loss = ((q_values - target) ** 2).mean()
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
            loss = ((q_values - target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
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
        # Process each transition to ensure states have shape (seq_length, state_dim)
        processed_batch = []
        for state, action, reward, next_state, done in batch:
            state = self._ensure_history(state)
            next_state = self._ensure_history(next_state)
            processed_batch.append((state, action, reward, next_state, done))

        # Unpack the processed transitions
        states, actions, rewards, next_states, dones = zip(*processed_batch)

        # Stack the individual states into a single tensor with shape:
        # (batch, seq_length, state_dim)
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()

        # Use mixed precision training if available
        if self.scaler is not None:
            with torch.amp.autocast():
                # Forward pass: Get Q-values for current states
                q_values, _ = self.q_net(states)
                q_values = q_values.gather(1, actions)  # Select Q-values for taken actions

                # Target network: Get Q-values for next states and compute target values
                with torch.no_grad():
                    max_next_q, _ = self.target_net(next_states)
                    max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                    target = rewards + self.gamma * max_next_q * (1 - dones)

                # Compute mean squared TD error loss
                loss = ((q_values - target) ** 2).mean()

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training fallback
            q_values, _ = self.q_net(states)
            q_values = q_values.gather(1, actions)
            with torch.no_grad():
                max_next_q, _ = self.target_net(next_states)
                max_next_q = max_next_q.max(1)[0].unsqueeze(1)
                target = rewards + self.gamma * max_next_q * (1 - dones)
            loss = ((q_values - target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Increment step count and update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Update the learning rate scheduler based on the current loss
        self.scheduler.step(loss)

        # Decay the exploration rate (epsilon)
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
