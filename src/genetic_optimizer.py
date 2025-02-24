"""
genetic_optimizer.py
--------------------
This module implements a genetic algorithm (GA) optimizer for trading strategy parameters.
It uses the DEAP library to evolve a population of individuals (parameter sets) and evaluates
them by running reinforcement learning (RL) training on each individual.
It also supports distributed evaluation using Ray or local multiprocessing.
"""

import gc
import json
import logging
import os
import random
import sys
import hashlib

import numpy as np
import pandas as pd
from deap import base, creator, tools
import ray
from multiprocessing import Pool
from src.config_loader import Config
from src.data_loader import DataLoader

# Set up logging for the GeneticOptimizer.
logger = logging.getLogger('GeneticOptimizer')
logger.setLevel(logging.DEBUG)

# Add a console handler so messages are printed to standard output.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optionally add a named pipe handler for logging.
pipe_path = '/tmp/genetic_optimizer_logpipe'
if not os.path.exists(pipe_path):
    os.mkfifo(pipe_path)
try:
    pipe = open(pipe_path, 'w')
    pipe_handler = logging.StreamHandler(pipe)
    pipe_handler.setLevel(logging.DEBUG)
    pipe_handler.setFormatter(formatter)
    logger.addHandler(pipe_handler)
except Exception as e:
    logger.error(f"Failed to open named pipe {pipe_path}: {e}")

@ray.remote
def ray_evaluate_individual(pickled_class, individual):
    """
    Remote function for evaluating one individual in the GA using Ray.
    Simply calls evaluate_individual on the provided GeneticOptimizer instance.
    
    Parameters:
        pickled_class: An instance of GeneticOptimizer.
        individual: A list of parameter values.
    
    Returns:
        A tuple (fitness, avg_profit, agent).
    """
    return pickled_class.evaluate_individual(individual)

def local_evaluate_individual(args):
    """
    Evaluate a single individual locally using multiprocessing.

    Parameters:
        args: A tuple containing (optimizer, individual).

    Returns:
        The result of evaluating the individual.
    """
    optimizer, individual = args
    return optimizer.evaluate_individual(individual)

class GeneticOptimizer:
    def __init__(self, data_loader: DataLoader, session_id=None, gen=None,
                 indicators_dir='precalculated_indicators_parquet',
                 checkpoint_dir='checkpoints', checkpoint_file=None):
        """
        Initialize the GeneticOptimizer.

        Parameters:
            data_loader (DataLoader): Instance for loading and processing market data.
            session_id (str): Unique identifier for the GA session.
            gen (int): Generation number to load population from; if None, start fresh.
            indicators_dir (str): Directory for pre-calculated indicators.
            checkpoint_dir (str): Directory for saving checkpoints.
            checkpoint_file (str): Optional checkpoint file to resume training.
        """
        self.data_loader = data_loader
        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists
        self.config = Config()  # Load configuration settings

        # Save a unique session id.
        self.session_id = session_id if session_id else "unknown-session"
        self.gen = gen

        # Define the technical indicators and parameter ranges.
        self.indicators = self.define_indicators()
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9)
        }
        # Create a mapping for parameters to indices.
        self.parameter_indices = self.create_parameter_indices()
        # Prepare base price data.
        self.prepare_data()

        # Setup DEAP (genetic algorithm framework).
        self.setup_deap()

        # Optionally, load a checkpoint to resume a previous run.
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def define_indicators(self):
        """
        Define the technical indicators and their parameter ranges.

        Returns:
            dict: Dictionary of indicators.
        """
        # For example, here we define VWAP, VWMA, VPVR, RSI, MACD with specific parameter ranges.
        return {
            'vwap': {'offset': (0, 50)},
            'vwma': {'length': (5, 200)},
            'vpvr': {
                '1min': {'width': (1000, 50000)},
                '5min': {'width': (200, 10000)},
                '15min': {'width': (100, 3500)},
                '30min': {'width': (50, 2000)},
                '1h': {'width': (30, 1000)},
                '4h': {'width': (10, 250)},
                '1d': {'width': (10, 60)},
            },
            'rsi': {'length': (5, 30)},
            'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
            'bbands': {'length': (10, 50), 'std_dev': (1.0, 3.0)},  # Bollinger Bands
            'stoch': {'k': (5, 20), 'd': (3, 10)},  # Stochastic Oscillator
            'atr': {'length': (5, 30)},  # Average True Range
            'obv': {}  # On-Balance Volume (no parameters needed)
        }

    def create_parameter_indices(self):
        """
        Create a mapping of each parameter to a unique index in an individual.

        Returns:
            dict: Mapping with keys as parameter tuples and values as indices.
        """
        parameter_indices = {}
        idx = 0
        # Define the supported timeframes.
        self.timeframes = ['1min', '5min', '15min', '30min', '1h', '4h', '1d']
        for indicator_name, data in self.indicators.items():
            if isinstance(data, dict):
                # Check if the keys in the indicator configuration are timeframes.
                if set(data.keys()).issubset(set(self.timeframes)):
                    for tf, param_dict in data.items():
                        for param_name, (low, high) in param_dict.items():
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
                else:
                    # If parameters are not separated by timeframe, assign them to all timeframes.
                    for param_name, (low, high) in data.items():
                        for tf in self.timeframes:
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
            else:
                raise ValueError(f"Indicators config for '{indicator_name}' must be a dict, got {type(data)}")
        # Add model-specific parameters.
        for param_name in self.model_params.keys():
            parameter_indices[('model', param_name)] = idx
            idx += 1
        return parameter_indices

    def prepare_data(self):
        """
        Load and save the base price data for the primary timeframe.
        """
        base_tf = self.data_loader.base_timeframe
        self.base_price_data = self.data_loader.tick_data[base_tf].copy()  # Create a copy for safety

    def setup_deap(self):
        """
        Initialize the DEAP framework by creating the Individual and Fitness classes,
        and registering genetic operators.
        """
        # Create a fitness class that we want to minimize.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Create an Individual class which is a list with a fitness attribute.
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        # Register the function to create an individual.
        self.toolbox.register("individual", self.init_ind, creator.Individual)
        # Register a population generator.
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # Register crossover, mutation, and selection operators.
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def init_ind(self, icls):
        """
        Create a single individual with random parameter values.

        Parameters:
            icls: The Individual class from DEAP.

        Returns:
            An instance of the individual.
        """
        keys = list(self.parameter_indices.keys())
        ind = []
        # Iterate over each parameter and choose a random value in the range.
        for k in keys:
            if k[0] == 'model':
                # For model parameters (e.g., threshold_buy), get the defined range.
                low, high = self.model_params[k[1]]
            else:
                # For indicators, extract the parameter range for the given timeframe.
                indicator_name, param_name, timeframe = k
                indicator_dict = self.indicators[indicator_name]
                if set(indicator_dict.keys()).issubset(set(self.timeframes)):
                    low, high = indicator_dict[timeframe][param_name]
                else:
                    low, high = indicator_dict[param_name]
            # Use randint if the range is integer, else use uniform for floats.
            val = random.randint(low, high) if isinstance(low, int) else random.uniform(low, high)
            ind.append(val)
        return icls(ind)

    def evaluate_individual(self, individual, useLocal=False):
        """
        Evaluate an individual by:
          1. Converting the individual's parameter list into a configuration.
          2. Computing technical indicators based on that configuration.
          3. Splitting the dataset into chunks and running RL training on each chunk in random order.

        Parameters:
            individual: A list of parameter values representing a candidate solution.
            useLocal (bool): If True, attempt to load precomputed indicator features from file.

        Returns:
            Tuple (fitness, avg_profit, agent), where fitness is negative average profit (to be minimized).
        """
        # Build configuration dictionary from the individual's parameters.
        config = self.extract_config_from_individual(individual)
        if useLocal:
            hash_str = hashlib.md5(str(individual).encode()).hexdigest()
            features_file = f"features/{hash_str}.parquet"
            if os.path.exists(features_file):
                logger.info("Loading precomputed indicator features from file...")
                features_df = pd.read_parquet(features_file)
            else:
                logger.info("Computing indicator features for individual (local mode)...")
                indicators = self.load_indicators(config)
                features_df = self.prepare_features(indicators)
                features_df.to_parquet(features_file)
        else:
            logger.info("Computing indicator features for individual...")
            indicators = self.load_indicators(config)
            features_df = self.prepare_features(indicators)

        features_df = self.data_loader.filter_data_by_date(features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )

        if len(features_df) < 100:
            logger.warning("Not enough data to run RL.")
            return (9999999.0,), 0.0, None

        if 'close' not in features_df.columns:
            logger.error("'close' column missing in features_df.")
            return (9999999.0,), 0.0, None

        # Extract price and indicator data.
        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')
        del features_df
        # Create the trading environment.
        env = self.create_environment(price_data, indicators_only)
        del indicators_only
        # Run RL training on the environment.
        logger.info("Starting RL training for individual using dataset-level chunking...")
        # Use our updated RL training function that chunks the dataset.
        agent, avg_profit = self.run_rl_training(env, episodes=self.config.get('episodes', 20))
        # Return negative average profit as fitness (since GA minimizes fitness).
        return (-avg_profit,), avg_profit, agent

    def extract_config_from_individual(self, individual):
        """
        Build a configuration dictionary from an individual's parameter values.

        Parameters:
            individual: List of parameter values.

        Returns:
            dict: Dictionary containing 'indicator_params' and 'model_params'.
        """
        config = {}
        indicator_params = {}
        model_params = {}
        keys = list(self.parameter_indices.keys())
        # Map each value to its corresponding parameter key.
        for i, val in enumerate(individual):
            key = keys[i]
            if key[0] == 'model':
                model_params[key[1]] = val
            else:
                indicator_name, param_name, timeframe = key
                if indicator_name not in indicator_params:
                    indicator_params[indicator_name] = {}
                if timeframe not in indicator_params[indicator_name]:
                    indicator_params[indicator_name][timeframe] = {}
                indicator_params[indicator_name][timeframe][param_name] = val
        config['indicator_params'] = indicator_params
        config['model_params'] = model_params
        return config

    def load_indicators(self, config):
        """
        Compute technical indicator DataFrames based on the configuration.

        Parameters:
            config (dict): Contains indicator parameters.

        Returns:
            dict: Dictionary mapping indicator names to their DataFrames.
        """
        indicator_params = config['indicator_params']
        indicators = {}
        # For each indicator, compute the DataFrame for each timeframe.
        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}
            for timeframe, params in timeframes_params.items():
                indicator_df = self.data_loader.calculate_indicator(indicator_name, params, timeframe)
                # If timeframe is not 1min, adjust the index so that data aligns with 1min bars.
                if timeframe != '1min':
                    shift_duration = pd.to_timedelta(timeframe)
                    indicator_df_shifted = indicator_df.copy()
                    indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                    indicator_df_1min = indicator_df_shifted.resample('1min').ffill()
                    indicator_df_1min.dropna(inplace=True)
                    indicator_df = indicator_df_1min
                indicators[indicator_name][timeframe] = indicator_df
        return indicators

    def prepare_features(self, indicators):
        """
        Concatenate the base price data with all indicator DataFrames to create a unified features DataFrame.

        Parameters:
            indicators (dict): Dictionary of indicator DataFrames.

        Returns:
            pd.DataFrame: Features DataFrame with 1-minute resolution.
        """
        # Start with the base price data.
        features_df = self.base_price_data.copy()
        # Loop over each indicator and join its DataFrame.
        for indicator_name, tf_dict in indicators.items():
            for timeframe, df in tf_dict.items():
                # Remove any unwanted characters from column names.
                df.columns = [col.replace('\n', '').strip() for col in df.columns]
                # Align the indicator DataFrame with the base data index.
                df = df.reindex(features_df.index, method='ffill')
                # Append a suffix so the column name includes the indicator name and timeframe.
                df = df.add_suffix(f'_{indicator_name}_{timeframe}')
                # Join the indicator data to the features.
                features_df = features_df.join(df)
        # Drop any rows with missing values.
        features_df.dropna(inplace=True)
        # Filter the DataFrame based on the simulation start and end dates from the config.
        filtered = self.data_loader.filter_data_by_date(
            features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )
        return filtered

    def create_environment(self, price_data, indicators):
        """
        Create a TradingEnvironment instance using price data and indicator data.

        Parameters:
            price_data (pd.DataFrame): Price data DataFrame.
            indicators (pd.DataFrame): Indicator data DataFrame.

        Returns:
            TradingEnvironment: A new trading environment.
        """
        from src.rl_environment import TradingEnvironment
        initial_capital = 100000  # Starting capital
        transaction_cost = 0.005  # Transaction fee percentage
        mode = self.config.get('training_mode')
        # Create and return the environment instance.
        return TradingEnvironment(price_data, indicators, initial_capital=initial_capital,
                                  transaction_cost=transaction_cost, mode=mode)

    def run_rl_training(self, env, episodes=1):
        """
        Run reinforcement learning training on the trading environment using dataset-level chunking.
        For each episode, the entire dataset (env.data) is split into chunks (of rows) of size chunk_size.
        The list of chunks is shuffled and the agent is trained on each chunk sequentially.
        This process is repeated for the specified number of episodes.

        Parameters:
            env: The trading environment instance.
            episodes (int): Number of complete passes (episodes) over the dataset chunks.

        Returns:
            tuple: (agent, avg_reward) where agent is the trained RL agent and
                   avg_reward is the average reward across episodes.
        """
        from src.rl_agent import DQNAgent
        import gc
        import time
        seq_length = self.config.get('seq_length', 720)
        chunk_size = self.config.get('chunk_size', 4000)
        batch_frequency = 32

        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3, seq_length=seq_length, epsilon_decay=0.995)
        agent.env = env  # Enable access to env.gain_loss and env.inventory
        full_data_np = env.data.values
        all_indices = np.arange(len(full_data_np))
        total_reward_sum = 0.0
        episode_count = 0
        transition_buffer = []

        for ep in range(episodes):
            num_chunks = int(np.ceil(len(all_indices) / chunk_size))
            chunk_indices = [all_indices[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
            np.random.shuffle(chunk_indices)
            ep_reward = 0.0

            for i, indices in enumerate(chunk_indices):
                chunk_len = len(indices)
                logger.info(f"Episode {ep + 1}: Processing chunk {i + 1}/{num_chunks} (size {chunk_len} rows)...")
                env.data_values = full_data_np[indices]
                env.n_steps = len(indices)
                env.timestamps_list = env.data.index[indices].tolist()
                state = env.reset()
                chunk_reward = 0.0
                done = False
                step_count = 0
                chunk_start_time = time.time()

                while not done:
                    action_start = time.time()
                    action = agent.select_action(state)
                    action_time = time.time() - action_start
                    next_state, reward, done, info = env.step(action)
                    transition_buffer.append((state, action, reward, next_state, done))
                    state = next_state
                    chunk_reward += reward
                    step_count += 1
                    if len(transition_buffer) >= batch_frequency or done:
                        agent.update_policy_from_batch(transition_buffer)
                        transition_buffer.clear()
                        gc.collect()
                    if step_count % 1000 == 0:
                        logger.debug(f"Step {step_count}: Action time={action_time:.3f}s")
                if step_count < chunk_len:
                    penalty_factory = (chunk_len - step_count) / chunk_len
                    chunk_reward *= (1 + penalty_factory)
                ep_reward += chunk_reward
                logger.info(f"Episode {ep + 1}, chunk {i + 1} completed. Chunk reward: {chunk_reward:.2f} "
                            f"(last step: {info.get('n_step', 0)})")
                logger.info(f"Chunk {i + 1} took {time.time() - chunk_start_time:.2f}s")
            total_reward_sum += ep_reward
            episode_count += 1
            logger.info(f"Episode {ep + 1} completed. Total episode reward: {ep_reward:.2f}")

        avg_reward = total_reward_sum / episode_count if episode_count > 0 else 0.0
        logger.info(f"RL training over {episodes} episodes completed. Average reward: {avg_reward:.2f}")
        del full_data_np
        gc.collect()
        return agent, avg_reward

    def evaluate_individuals(self, individuals):
        """
        Evaluate a list of individuals (parameter sets) using Ray or local multiprocessing.

        Parameters:
            individuals (list): List of individuals.

        Returns:
            list: List of evaluation results.
        """
        processing = self.config.get("processing", "ray").lower()
        if processing == "ray":
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            # Use Ray to evaluate each individual in parallel.
            tasks = [ray_evaluate_individual.remote(self, ind) for ind in individuals]
            results = ray.get(tasks)
        elif processing == "local":
            # Use local multiprocessing to evaluate individuals.
            with Pool(processes=self.config.get("num_processes", 4)) as pool:
                results = pool.map(local_evaluate_individual, [(self, ind) for ind in individuals])
        else:
            # Fallback: evaluate sequentially.
            results = [self.evaluate_individual(ind) for ind in individuals]
        return results

    def run(self):
        """
        Main method to run the genetic algorithm optimization.
        Evolves a population over many generations and saves the best performing RL agent(s).
        """
        NGEN = 100        # Number of generations
        CXPB = 0.5        # Crossover probability
        MUTPB = 0.1       # Mutation probability
        INITIAL_POPULATION = 100

        # Ensure directories for saving population and weights exist.
        os.makedirs(f"population/{self.session_id}", exist_ok=True)
        os.makedirs(f"weights/{self.session_id}", exist_ok=True)

        # If a generation is specified, load that population; otherwise, create a new one.
        if self.gen is not None:
            pop = self.load_population(self.session_id, self.gen)
            if not pop:
                pop = self.toolbox.population(n=INITIAL_POPULATION)
        else:
            pop = self.toolbox.population(n=INITIAL_POPULATION)

        logger.info("Evaluating initial population...")
        results = self.evaluate_individuals(pop)
        # Assign fitness, average profit, and agent to each individual.
        for ind, (fit, avg_profit, agent) in zip(pop, results):
            ind.fitness.values = fit
            ind.avg_profit = avg_profit
            ind.agent = agent

        # Start the evolution process over generations.
        for gen in range(1, NGEN + 1):
            logger.info(f"=== Generation {gen} ===")
            # Increase population size after the first generation.
            desired_pop_size = 1000 if gen > 1 else INITIAL_POPULATION
            offspring = self.toolbox.select(pop, desired_pop_size)
            # Clone the offspring to avoid altering originals.
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover.
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            # Apply mutation.
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate all individuals that have lost a valid fitness.
            invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Evaluating {len(invalid_inds)} individuals with {self.config.get('processing', 'ray')} processing...")
            results = self.evaluate_individuals(invalid_inds)
            for ind, (fit, avg_profit, agent) in zip(invalid_inds, results):
                ind.fitness.values = fit
                ind.avg_profit = avg_profit
                ind.agent = agent
            # Update the population.
            pop[:] = offspring

            # Save the current population to a JSON file.
            pop_file = f"population/{self.session_id}/gen{gen}.json"
            population_data = []
            for i, ind in enumerate(pop):
                population_data.append({
                    'individual_index': i,
                    'parameters': list(ind),
                    'fitness': ind.fitness.values[0],
                    'avg_profit': getattr(ind, 'avg_profit', None)
                })
            with open(pop_file, 'w') as f:
                json.dump(population_data, f, indent=2)

            # Find the best individual.
            best_ind = tools.selBest(pop, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            best_reward = getattr(best_ind, 'avg_profit', None)
            best_agent = getattr(best_ind, 'agent', None)
            logger.info(f"Best Fitness: {best_fitness} | Best Reward: {best_reward}")

            # Save the RL agent's weights if it has achieved a high profit.
            for ind_id, ind in enumerate(pop):
                agent_to_save = getattr(ind, 'agent', None)
                ind_profit = getattr(ind, 'avg_profit', 0)
                if agent_to_save is not None and ind_profit > 100000:
                    use_id = str(ind_id)
                    use_profit = int(ind_profit)
                    weight_file = f"weights/{self.session_id}/gen{gen}-{use_id}-{use_profit}.pth"
                    agent_to_save.save(weight_file)
                    logger.info(f"Saved RL agent's weights to {weight_file} because profit {ind_profit} > 100000")
            # Clean up agent objects to save memory.
            for ind in pop:
                if hasattr(ind, 'agent'):
                    del ind.agent
            gc.collect()
            logger.info(f"Generation {gen} completed. Best fitness: {best_fitness} | Best reward: {best_reward}")

        logger.info("DEAP Optimization Completed")

    def load_population(self, session_id: str, generation_number: int):
        """
        Load a saved population from a JSON file.

        Parameters:
            session_id (str): Session identifier.
            generation_number (int): Generation number to load.

        Returns:
            list: List of individuals.
        """
        pop_file = f"population/{session_id}/gen{generation_number}.json"
        if not os.path.exists(pop_file):
            logger.warning(f"Population file {pop_file} not found.")
            return []
        with open(pop_file, 'r') as f:
            population_data = json.load(f)
        pop = []
        for item in population_data:
            params = item['parameters']
            ind = creator.Individual(params)
            pop.append(ind)
        logger.info(f"Loaded population of size {len(pop)} from {pop_file}")
        return pop

    def load_checkpoint(self, checkpoint_file):
        """
        Stub for loading a checkpoint.
        (Not implemented here.)
        """
        pass

    def test_individual(self, individual_params=None):
        """
        Stub for testing a single individual.
        (Not implemented here.)
        """
        pass

    def debug_single_individual(self, individual_params=None):
        """
        Debug a single individual without running the full GA loop.

        Parameters:
            individual_params: Optional list of parameter values.

        Returns:
            Tuple (fitness, avg_profit, agent).
        """
        from deap import creator
        if individual_params is None:
            keys = list(self.parameter_indices.keys())
            individual_params = []
            # Generate random parameters for each key.
            for k in keys:
                if k[0] == 'model':
                    low, high = self.model_params[k[1]]
                else:
                    indicator_name, param_name, timeframe = k
                    indicator_dict = self.indicators[indicator_name]
                    if set(indicator_dict.keys()).issubset(set(self.timeframes)):
                        low, high = indicator_dict[timeframe][param_name]
                    else:
                        low, high = indicator_dict[param_name]
                val = random.randint(low, high) if isinstance(low, int) else random.uniform(low, high)
                individual_params.append(val)
        ind = creator.Individual(individual_params)
        logger.info("=== Debugging single individual ===")
        fit, avg_profit, agent = self.evaluate_individual(ind, useLocal=True)
        logger.info(f"Evaluation finished. Fit: {fit}, Profit: {avg_profit}")
        if agent is not None:
            logger.info("An RL agent was created; you can place breakpoints inside run_rl_training or environment.")
        else:
            logger.info("No RL agent was created (possibly not enough data).")
        return fit, avg_profit, agent
