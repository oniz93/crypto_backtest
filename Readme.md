# Bitmex Liquidation Project

This project is a trading strategy optimization framework that combines genetic algorithms (GA) with reinforcement learning (RL) to optimize and evaluate trading strategies. The project processes historical market data, calculates technical indicators, and then uses a GA to evolve a population of candidate parameter sets. Each candidate is evaluated by training an RL agent in a simulated trading environment. The overall goal is to find the best set of parameters for the trading strategy.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Technical Details](#technical-details)
  - [Data Processing](#data-processing)
  - [Technical Indicators](#technical-indicators)
  - [Genetic Algorithm Optimization](#genetic-algorithm-optimization)
  - [Reinforcement Learning Agent](#reinforcement-learning-agent)
  - [Trading Environment Simulation](#trading-environment-simulation)
- [Deployment](#deployment)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Overview

This project implements a complete pipeline for optimizing a trading strategy:

1. **Data Loading and Preprocessing**:  
   Historical tick data is loaded from Parquet files and resampled into various timeframes (e.g. 1min, 5min, 15min). Technical indicators (like SMA, EMA, RSI, MACD, etc.) are computed on the market data using the `pandas_ta` library and are sometimes accelerated using Numba.

2. **Technical Indicator Precalculation**:  
   A separate script (`precalculate_indicators.py`) calculates and saves the indicators for later use. This speeds up evaluations during the GA process.

3. **Genetic Algorithm Optimization**:  
   The genetic algorithm (implemented with DEAP in `genetic_optimizer.py`) evolves a population of candidate parameter sets. Each individual represents a set of parameters (for technical indicators and model settings). The GA evaluates each candidate by running an RL training process using the specified parameters.

4. **Reinforcement Learning**:  
   An RL agent (implemented in `rl_agent.py`) is used to simulate trading in a custom environment (`rl_environment.py`). The RL agent is built with a combined recurrent network (LSTM + GRU) and is trained on historical data. The profit achieved by the agent is used as the fitness measure for the GA.

5. **Trading Simulation**:  
   The `simulation.py` module provides a high-level interface to run the simulation by simply invoking the strategy’s `calculate_profit()` method.

6. **Deployment and Data Merging**:  
   Additional scripts are provided for merging CSV files (trades, funding history, and different timeframes) into Parquet files. A shell script (`syncall.sh`) is included to deploy the project to multiple hosts.

## Project Structure

```
.
├── config.yaml                 # Main configuration file.
├── convert_trades.py           # Script to convert CSV trade files to Parquet.
├── indicators_config.json      # JSON configuration for technical indicators.
├── main.py                     # Main entry point for the GA optimization.
├── merge_funding_history.py    # Merge funding history CSVs into one Parquet file.
├── merge_timeframe_files.py    # Merge tick data CSVs for different timeframes.
├── merge_trades.py             # Merge aggregated trade CSVs into one Parquet file.
├── precalculate_indicators.py  # Precalculate and save technical indicators.
├── syncall.sh                  # Shell script to deploy files to remote hosts.
├── test_torch.py               # Test script for checking Torch backend (MPS/CUDA).
├── train_rl.py                 # Script to train the RL agent.
├── use_trained_model.py        # Script for using a trained model for inference.
└── src/                        # Source code folder.
    ├── __init__.py             # Package initializer.
    ├── config_loader.py        # Loads configuration settings from YAML.
    ├── data_loader.py          # Loads and processes market data; computes indicators.
    ├── genetic_optimizer.py    # Implements the GA optimizer using DEAP (and Ray/multiprocessing).
    ├── rl_agent.py             # Defines the RL agent with a combined LSTM+GRU network.
    ├── rl_environment.py       # Simulated trading environment for RL training.
    └── trading_strategy.py     # Implements a trading strategy simulation.
```

## Workflow

1. **Start with `main.py`**  
   - The program starts here by parsing command-line arguments (e.g. session ID and generation).
   - It creates a `DataLoader` instance that imports and resamples tick data.
   - The `GeneticOptimizer` is then instantiated with the loaded data.

2. **Genetic Algorithm Optimization**  
   - Inside `GeneticOptimizer`, technical indicators are defined and a mapping from parameters to indices is created.
   - The GA creates an initial population of candidate parameter sets.
   - Each individual is evaluated by:
     - Converting the candidate parameters into a configuration.
     - Using the `DataLoader` to compute technical indicators based on the configuration.
     - Preparing a features DataFrame by merging price data and indicator data.
     - Creating a trading environment with these features.
     - Running RL training with a `DQNAgent` in the environment to compute profit.
   - The fitness (negative profit, since the GA minimizes fitness) is assigned to each individual.
   - The GA then evolves the population over many generations using crossover, mutation, and selection.
   - Optionally, the best-performing RL agent's weights are saved for later use.

3. **Reinforcement Learning Training**  
   - The RL agent (in `rl_agent.py`) uses a combined LSTM+GRU network to process a history of timesteps (as specified by `seq_length`).
   - The agent interacts with the trading environment (`rl_environment.py`), where each step simulates a trade based on the agent’s chosen action.
   - Transitions (state, action, reward, next_state) are stored in a replay buffer, and the network is updated via gradient descent.

4. **Simulation and Evaluation**  
   - The `simulation.py` module provides a simple interface to run a strategy’s simulation (for example, the one implemented by the GA or RL training).
   - Other scripts (like `use_trained_model.py`) can be used for inference with a trained model.

5. **Data Preprocessing and Deployment**  
   - Scripts like `convert_trades.py`, `merge_funding_history.py`, and `merge_timeframe_files.py` are used to preprocess data.
   - The `precalculate_indicators.py` script precalculates technical indicators for faster evaluations.
   - The `syncall.sh` script is provided to deploy the project to multiple remote hosts.

## Technical Details

### Data Processing

- **DataLoader (`src/data_loader.py`)**  
  Loads tick data from Parquet files, resamples it into different timeframes, and calculates technical indicators.  
  Uses Numba-accelerated functions for computing cumulative volume profiles to speed up performance.

- **Filtering and Resampling**  
  Data is filtered by date (using cutoff dates specified in `config.yaml`) and resampled using standard aggregation functions (first, max, min, last, sum).

### Technical Indicators

- **Indicators**  
  Defined in `indicators_config.json` and processed by functions in `data_loader.py`.  
  Indicators include SMA, EMA, RSI, MACD, ATR, VPVR, etc.  
  Timing logs can be enabled (via config) to help identify slow computations.

### Genetic Algorithm Optimization

- **GeneticOptimizer (`src/genetic_optimizer.py`)**  
  Uses DEAP to create and evolve a population of individuals (candidate parameter sets).  
  Individuals are evaluated by running RL training with the parameters specified by the individual.  
  Distributed evaluation is supported via Ray or local multiprocessing.

### Reinforcement Learning Agent

- **RL Agent (`src/rl_agent.py`)**  
  Implements a DQN agent with GRU layers.  
  The network processes a history of timesteps (e.g., 1440) and outputs Q-values for actions (hold, buy, sell).  
  The agent maintains a replay buffer and periodically updates a target network.

### Trading Environment Simulation

- **TradingEnvironment (`src/rl_environment.py`)**  
  Simulates the trading process by merging price and indicator data.  
  Computes rewards based on portfolio changes after executing buy/sell/hold actions.  
  Provides state vectors that include normalized market features and extra features (e.g., cash ratio, inventory).

## Deployment

- **Syncall Script (`syncall.sh`)**  
  Deploys project files to remote Raspberry Pi hosts in parallel using rsync.

## Dependencies

The project requires the following Python packages (see `requirements.txt`):

- numpy
- torch
- pandas
- pandas_ta
- PyYAML
- deap
- ray
- pyarrow
- numba
- numexpr

## Usage

1. **Preprocessing Data and Indicators**  
   - Run `convert_trades.py` to convert CSV trade data to Parquet.  
   - Run `merge_funding_history.py` and `merge_timeframe_files.py` to process other data files.  
   - Run `precalculate_indicators.py` to precalculate technical indicators.

2. **Running the Genetic Algorithm**  
   - Execute `main.py` to start the genetic optimization process.  
     ```bash
     python main.py --session_id 123456 --gen 1
     ```
   - The GA will evolve individuals over many generations and save the best-performing RL agent weights.

3. **Training the RL Agent Separately**  
   - Run `train_rl.py` to train the RL agent on the trading environment.

4. **Using the Trained Model**  
   - Run `use_trained_model.py` to load the trained model and perform inference.

5. **Deployment**  
   - Use `syncall.sh` to deploy the project files to remote hosts.

### Usage with Ray

To start ray on the master
```
ray start --head --port=6379 --num-cpus=15 --node-ip-address='192.168.1.221'
```

To start ray on the nodes
```
PYTHONPATH="$PYTHONPATH:$PWD" RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address='192.168.1.221:6379'
```