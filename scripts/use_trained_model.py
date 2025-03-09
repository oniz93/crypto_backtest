"""
use_trained_model.py
--------------------
This script loads a trained Q-network model and uses it for action selection.
It demonstrates how to perform inference with the RL model on new market data.
"""

import numpy as np
import torch
from src.data_loader import DataLoader
from src.rl_agent import QNetwork

# These should match the training configuration.
state_dim = 10  # Adjust to match your training setting.
action_dim = 3  # For example: 0 = hold, 1 = buy, 2 = sell.

# Initialize the network architecture.
model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load('dqn_model.pth', map_location=torch.device('cpu')))
model.eval()


def select_greedy_action(model, state):
    """
    Select an action greedily (no exploration).

    Parameters:
        model: The Q-network.
        state: Input state (1D array).

    Returns:
        int: Chosen action.
    """
    with torch.no_grad():
        q_values = model(torch.FloatTensor(state).unsqueeze(0))
    return q_values.argmax().item()


# Load market data.
data_loader = DataLoader()
data_loader.import_ticks()
data_loader.resample_data()

base_tf = data_loader.base_timeframe
price_data = data_loader.tick_data[base_tf].copy()
if 'close' not in price_data.columns:
    raise ValueError("Close price not found in price_data")

# For demonstration, create a dummy state vector from the last row.
indicator_values = price_data.iloc[-1].values  # Last row of data.
inventory = 0
cash_ratio = 1.0
state = np.concatenate([indicator_values, [inventory, cash_ratio]])

action = select_greedy_action(model, state)
if action == 0:
    print("Action: HOLD")
elif action == 1:
    print("Action: BUY")
elif action == 2:
    print("Action: SELL")
