"""
genetic_optimizer.py
--------------------
This module implements a genetic algorithm (GA) optimizer for trading strategy parameters.
It uses the DEAP library to evolve a population of individuals (parameter sets) and evaluates
them by running reinforcement learning (RL) training on each individual.
It also supports distributed evaluation using Ray or local multiprocessing.
It now uses a conditional dataframe library (cuDF if CUDA is available, else pandas).
"""

import gc
import json
import logging
import os
import random
import sys
import hashlib
import time

import numpy as np
import pandas as pd2
import cudf
import cupy as cp
import dask
# -----------------------------------------------------------------------------
# CONDITIONAL DATAFRAME LIBRARY IMPORT:
# Use cuDF when CUDA is available, otherwise fallback to pandas.
# -----------------------------------------------------------------------------

USING_CUDF = False
NUM_GPU = 0

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() > 0:
        import cudf as pd
        import dask_cudf  # Add dask_cudf import
        USING_CUDF = True
        NUM_GPU = cp.cuda.runtime.getDeviceCount()  # Ensure consistency
    else:
        import pandas as pd
        USING_CUDF = False
        NUM_GPU = 0
except Exception:
    import pandas as pd

from deap import base, creator, tools
import ray
from multiprocessing import Pool
from src.config_loader import Config
from src.data_loader import DataLoader
from dask.dataframe import DataFrame as DaskDataFrame
import dask.dataframe as dd
import dask_expr


import warnings
warnings.filterwarnings("ignore")

# Set up logging for the GeneticOptimizer.
logger = logging.getLogger('GeneticOptimizer')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
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
    Remote function for evaluating one individual using Ray.
    """
    return pickled_class.evaluate_individual(individual)

def local_evaluate_individual(args):
    """
    Evaluate a single individual locally using multiprocessing.
    """
    optimizer, individual = args
    return optimizer.evaluate_individual(individual)

class GeneticOptimizer:
    def __init__(self, data_loader: DataLoader, session_id=None, gen=None,
                 indicators_dir='precalculated_indicators_parquet',
                 checkpoint_dir='checkpoints', checkpoint_file=None):
        """
        Initialize the GeneticOptimizer.
        """
        self.data_loader = data_loader
        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.config = Config()
        self.session_id = session_id if session_id else "unknown-session"
        self.gen = gen
        self.indicators = self.define_indicators()
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9)
        }
        # Use the timeframes from the data loader to differentiate frequency strings.
        self.timeframes = self.data_loader.timeframes
        self.parameter_indices = self.create_parameter_indices()
        self.prepare_data()
        self.setup_deap()
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def define_indicators(self):
        """
        Define the technical indicators and their parameter ranges.
        """
        return {
            'vwap': {'offset': (0, 50)},
            'vwma': {'length': (5, 200)},
            'vpvr': {
                # Note: The keys here should match the entries in self.timeframes.
                # When using pandas, self.timeframes uses 1T/1D etc.
                '1T': {'width': (1000, 50000)},
                '5T': {'width': (200, 10000)},
                '15T': {'width': (100, 3500)},
                '30T': {'width': (50, 2000)},
                '1H': {'width': (30, 1000)},
                '4H': {'width': (10, 250)},
                '1D': {'width': (10, 60)},
            },
            'rsi': {'length': (5, 30)},
            'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
            'bbands': {'length': (10, 50), 'std_dev': (1.0, 3.0)},
            'stoch': {'k': (5, 20), 'd': (3, 10)},
            'atr': {'length': (5, 30)},
            'roc': {'length': (5, 30)},
            'hist_vol': {'length': (5, 50)},
            'obv': {}
        }

    def create_parameter_indices(self):
        """
        Create a mapping of each parameter to a unique index.
        """
        parameter_indices = {}
        idx = 0
        for indicator_name, data in self.indicators.items():
            if not isinstance(data, dict):
                raise ValueError(f"Indicators config for '{indicator_name}' must be a dict, got {type(data)}")
            # Skip if data is empty (e.g., 'obv')
            if not data:
                continue
            for tf in self.timeframes:
                if tf in data:
                    # Timeframe-specific parameters (e.g., 'vpvr')
                    param_dict = data[tf]
                    if not isinstance(param_dict, dict):
                        raise ValueError(f"Timeframe '{tf}' for '{indicator_name}' must be a dict, got {type(param_dict)}")
                    for param_name, limits in param_dict.items():
                        try:
                            low, high = limits
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
                        except ValueError:
                            raise ValueError(f"Indicator '{indicator_name}' timeframe '{tf}' param '{param_name}' has invalid limits: {limits}, expected (low, high)")
                else:
                    # Non-timeframe-specific parameters (e.g., 'vwap', 'rsi')
                    for param_name, limits in data.items():
                        try:
                            low, high = limits
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
                        except ValueError:
                            raise ValueError(f"Indicator '{indicator_name}' param '{param_name}' has invalid limits: {limits}, expected (low, high)")
        for param_name in self.model_params.keys():
            parameter_indices[('model', param_name)] = idx
            idx += 1
        return parameter_indices

    def prepare_data(self):
        """
        Prepare base price data.
        """
        base_tf = self.data_loader.base_timeframe
        self.base_price_data = self.data_loader.tick_data[base_tf].copy()

    def setup_deap(self):
        """
        Initialize the DEAP framework.
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.init_ind, creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def init_ind(self, icls):
        """
        Create a single individual with random parameter values.
        """
        keys = list(self.parameter_indices.keys())
        ind = []
        for k in keys:
            if k[0] == 'model':
                low, high = self.model_params[k[1]]
            else:
                indicator_name, param_name, timeframe = k
                indicator_dict = self.indicators[indicator_name]
                # Check if parameters are provided per timeframe
                if timeframe in indicator_dict:
                    low, high = indicator_dict[timeframe][param_name]
                else:
                    low, high = indicator_dict[param_name]
            val = random.randint(low, high) if isinstance(low, int) else random.uniform(low, high)
            ind.append(val)
        return icls(ind)

    def evaluate_individual(self, individual, useLocal=False):
        """
        Evaluate an individual by computing indicator features, filtering data,
        and running RL training.
        """
        config = self.extract_config_from_individual(individual)
        if useLocal:
            hash_str = hashlib.md5(str(individual).encode()).hexdigest()
            features_file = f"features/{hash_str}.parquet"
            if os.path.exists(features_file):
                logger.info("Loading precomputed indicator features from file...")
                features_df = pd.read_parquet(features_file)  # Always pandas for saved files
            else:
                logger.info("Computing indicator features for individual (local mode)...")
                indicators = self.load_indicators(config)
                features_df = self.prepare_features(indicators)
                features_df.to_parquet(features_file)
        else:
            logger.info("Computing indicator features for individual...")
            indicators = self.load_indicators(config)
            features_df = self.prepare_features(indicators)

        features_df = self.data_loader.filter_data_by_date(
            features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )

        if len(features_df) < 100:
            logger.warning("Not enough data to run RL.")
            return (9999999.0,), 0.0, None

        if 'close' not in features_df.columns:
            logger.error("'close' column missing in features_df.")
            return (9999999.0,), 0.0, None

        price_data = features_df[['close']]
        indicators_only = features_df.drop(columns=['close'], errors='ignore')
        del features_df
        env = self.create_environment(price_data, indicators_only)
        del indicators_only
        logger.info("Starting RL training for individual using dataset-level chunking...")
        agent, avg_profit = self.run_rl_training(env, episodes=self.config.get('episodes', 20))
        return (-avg_profit,), avg_profit, agent

    def extract_config_from_individual(self, individual):
        """
        Build a configuration dictionary from an individual's parameters.
        """
        config = {}
        indicator_params = {}
        model_params = {}
        keys = list(self.parameter_indices.keys())
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
    
    @staticmethod
    def _calculate_indicator_worker(args):
        """
        Helper worker function to compute a single indicator on a given timeframe.
        This function is intended to run in a separate process.
        
        Parameters:
            args (tuple): Contains (data_loader, indicator_name, timeframe, params)
            
        Returns:
            tuple: (indicator_name, timeframe, indicator_df)
        """
        data_loader, indicator_name, timeframe, params = args
        # Calculate the raw indicator using the DataLoader method.
        # If GPUs are available, data_loader.tick_data (and therefore the indicator computation)
        # will be using cuDF.

        action_start = time.time()
        logger.info(f"Calculating {indicator_name} on TF {timeframe} with parameters {params}")
        indicator_df = data_loader.calculate_indicator(indicator_name, params, timeframe)
        logger.info(f"Calculation done {indicator_name} on TF {timeframe} with parameters {params} took {time.time() - action_start:.2f}s")
        
        import pandas as pd2  # Used for timedelta conversion
        
        if USING_CUDF and NUM_GPU > 1:
            import dask_cudf
            # Convert the cuDF DataFrame into a dask_cudf DataFrame to leverage multi-GPU processing.
            indicator_df = dask_cudf.from_cudf(indicator_df, npartitions=NUM_GPU)
            if timeframe != data_loader.base_timeframe:
                shift_duration = pd2.to_timedelta(timeframe)
                # Shift the index on each partition without bringing the data to CPU.
                indicator_df = indicator_df.map_partitions(
                    lambda df: df.set_index(df.index - shift_duration)
                )
                # Use dask_cudf's native resample which runs on the GPUs.
                indicator_df = indicator_df.compute()
                indicator_df = indicator_df.resample(data_loader.base_timeframe).ffill()
                indicator_df = indicator_df.dropna()
        else:
            # Single GPU or CPU mode: use pandas resampling.
            if timeframe != data_loader.base_timeframe:
                shift_duration = pd2.to_timedelta(timeframe)
                indicator_df_shifted = indicator_df.copy()
                indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                indicator_df_1min = indicator_df_shifted.resample(data_loader.base_timeframe).ffill()
                indicator_df_1min.dropna(inplace=True)
                indicator_df = indicator_df_1min
        return (indicator_name, timeframe, indicator_df)
    
    
    def load_indicators(self, config):
        """
        Compute technical indicator DataFrames in parallel based on the configuration.
        Each indicator (for a given timeframe) is computed in a separate process.
        
        Returns:
            dict: A nested dictionary with indicator names as keys and dictionaries 
                  mapping each timeframe to its computed indicator DataFrame.
        """
        indicator_params = config['indicator_params']
        tasks = []
        indicators = {}

        # Create a list of tasks: one per indicator/timeframe combination.
        for indicator_name, timeframes_params in indicator_params.items():
            for timeframe, params in timeframes_params.items():
                tasks.append((self.data_loader, indicator_name, timeframe, params))
        
        # Use the number of processes specified in config.
        num_processes = config.get("num_processes", 5)
        from multiprocessing import Pool
        with Pool(processes=num_processes) as pool:
            results = pool.map(GeneticOptimizer._calculate_indicator_worker, tasks)
        
        # Combine the results into a nested dictionary.
        for indicator_name, timeframe, indicator_df in results:
            if indicator_name not in indicators:
                indicators[indicator_name] = {}
            indicators[indicator_name][timeframe] = indicator_df
        
        return indicators
    
    def prepare_features(self, indicators):
        """
        Concatenate the base price data with all indicator DataFrames.
        """
        # Initialize features_df based on GPU mode
        if USING_CUDF and NUM_GPU > 1:
            # Multi-GPU: Convert to Dask-cuDF
            features_df = dask_cudf.from_cudf(self.base_price_data, npartitions=NUM_GPU)
        else:
            # Single-GPU: Keep as cuDF
            features_df = (self.base_price_data.copy() 
                        if isinstance(self.base_price_data, cudf.DataFrame) 
                        else cudf.from_pandas(self.base_price_data))
        
        logger.info(f"DataFrame type for features_df: {type(features_df)}")

        # Process each indicator
        for indicator_name, tf_dict in indicators.items():
            for timeframe, df in tf_dict.items():
                action_start = time.time()
                logger.info(f"Merging {indicator_name} on TF {timeframe}")
                logger.info(f"DataFrame type for '{indicator_name}' '{timeframe}': {type(df)}")

                # Ensure df matches features_df type
                if USING_CUDF and NUM_GPU > 1:
                    if isinstance(df, (dask_cudf.DataFrame, dask_expr._collection.DataFrame)):
                        logger.info(f"Not converting")
                        # Already Dask-cuDF, no conversion needed
                        pass
                    elif isinstance(df, pd.DataFrame):  # cuDF DataFrame
                        logger.info(f"Converting from pd.DataFrame")
                        df = dask_cudf.from_cudf(df, npartitions=NUM_GPU)
                    elif isinstance(df, pd2.DataFrame):  # pandas DataFrame
                        logger.info(f"Converting from pd2.DataFrame")
                        df = dask_cudf.from_cudf(pd.from_pandas(df), npartitions=NUM_GPU)
                    elif isinstance(df, DaskDataFrame):  # Handle Dask DataFrame explicitly
                        logger.info(f"Converting from DaskDataFrame")
                        df = dask_cudf.from_cudf(df.compute(), npartitions=NUM_GPU)
                    elif hasattr(df, 'to_cudf'):  # Handle dask_expr.DataFrame or similar
                        logger.info(f"Converting from to_cudf")
                        df = dask_cudf.from_cudf(df.compute(), npartitions=NUM_GPU)
                else:
                    logger.info(f"Apply compute?")
                    # For local mode or single GPU, ensure it's a pandas or cuDF DataFrame
                    if hasattr(df, 'compute'):  # Handle dask_expr.DataFrame or dask_cudf.DataFrame
                        logger.info(f"Compute")
                        df = df.compute()  # Convert to concrete DataFrame (pandas or cuDF)
                    elif not isinstance(df, (pd.DataFrame, pd2.DataFrame)):
                        raise TypeError(f"Unexpected DataFrame type for '{indicator_name}' '{timeframe}': {type(df)}")
                # Clean column names
                df.columns = [col.replace('\n', '').strip() for col in df.columns]


                logger.info(f"DataFrame type after conversion for '{indicator_name}' '{timeframe}': {type(df)}")

                # Resample or reindex as needed
                if timeframe != self.data_loader.base_timeframe:
                    shift_duration = pd2.to_timedelta(timeframe)
                    df_shifted = df.copy()
                    df_shifted.index = df_shifted.index - shift_duration
                    if USING_CUDF and NUM_GPU > 1:
                        # For Dask-cuDF (multi-GPU), use map_partitions to resample and ffill
                        df = df_shifted.map_partitions(
                            lambda part: part.resample(self.data_loader.base_timeframe).first().ffill()
                        )
                        logger.info(f"DataFrame type after repartition for '{indicator_name}' '{timeframe}': {type(df)}")
                        df = dask_cudf.from_cudf(pd2.DataFrame(df.compute().to_numpy()), npartitions=NUM_GPU)
                    else:
                        # For cuDF or pandas (single-GPU or local), perform resample and ffill directly
                        df = df_shifted.resample(self.data_loader.base_timeframe).first().ffill()
                else:
                    # Reindex to match features_df.index
                    if USING_CUDF and NUM_GPU > 1:
                        target_index = (features_df.index.compute() 
                                    if isinstance(features_df.index, dask.array.Array) 
                                    else features_df.index)
                        logger.info(f"target_index type: {type(target_index)}")
                        target_index = target_index.to_series()
                        df = df.map_partitions(
                            lambda part: part.reindex(index=target_index, method='ffill')
                        )
                    else:
                        df = df.reindex(features_df.index, method='ffill')

                logger.info(f"DataFrame type before join for '{indicator_name}' '{timeframe}': {type(df)}")
                logger.info(f"features_df type before join: {type(df)}")
                # Perform the join
                features_df = features_df.join(df, how='inner')
                logger.info(f"Merging {indicator_name} on TF {timeframe} took {time.time() - action_start:.2f}s")

        # Compute RSI divergence (example logic, adjust as needed)
        rsi_1min_first_col = indicators['rsi'][self.data_loader.base_timeframe].columns[0]
        rsi_5min_first_col = indicators['rsi']['5T'].columns[0]
        rsi_1min = features_df[f'{rsi_1min_first_col}_rsi_{self.data_loader.base_timeframe}']
        rsi_5min = indicators['rsi']['5T'][rsi_5min_first_col].reindex(rsi_1min.index, method='ffill')
        features_df['RSI_Divergence'] = rsi_1min - rsi_5min

        # Drop NaN values
        features_df = features_df.dropna()

        # Filter by date
        filtered = self.data_loader.filter_data_by_date(
            features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )
        if USING_CUDF and NUM_GPU > 1:
            filtered = filtered.compute()  # Compute final result for multi-GPU
        return filtered

    def create_environment(self, price_data, indicators):
        """
        Create a TradingEnvironment instance using price and indicator data.
        """
        from src.rl_environment import TradingEnvironment
        initial_capital = 100000
        transaction_cost = 0.005
        mode = self.config.get('training_mode')
        return TradingEnvironment(price_data, indicators, initial_capital=initial_capital,
                                  transaction_cost=transaction_cost, mode=mode)

    def run_rl_training(self, env, episodes=1):
        """
        Run reinforcement learning training on the trading environment using dataset-level chunking.
        """
        from src.rl_agent import DQNAgent
        import gc
        import time
        seq_length = self.config.get('seq_length', 720)
        chunk_size = self.config.get('chunk_size', 4000)
        batch_frequency = 32

        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3, seq_length=seq_length, epsilon_decay=0.995)
        agent.env = env
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
                profit_sells = 0
                loss_sells = 0
                profit_total = 0.0
                loss_total = 0.0

                while not done:
                    action_start = time.time()
                    action = agent.select_action(state)
                    action_time = time.time() - action_start
                    next_state, reward, done, info = env.step(action)
                    transition_buffer.append((state.copy(), action, reward, next_state.copy(), done))
                    state = next_state
                    chunk_reward += reward
                    step_count += 1
                    is_sell = info.get('is_sell', 0)
                    if is_sell == 1:
                        profit_sells += 1
                        profit_total += info.get('gain_loss', 0.0)
                    elif is_sell == -1:
                        loss_sells += 1
                        loss_total += info.get('gain_loss', 0.0)

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
                logger.info(f"Episode {ep + 1}, chunk {i + 1} completed. Chunk reward: {chunk_reward:.2f} (last step: {info.get('n_step', 0)})")
                logger.info(f"Chunk {i + 1} took {time.time() - chunk_start_time:.2f}s")
                logger.info(f"Num profit sell {profit_sells} -> {profit_total:.2f}")
                logger.info(f"Num loss sell {loss_sells} -> {loss_total:.2f}")
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
        Evaluate a list of individuals using Ray or local multiprocessing.
        """
        processing = self.config.get("processing", "ray").lower()
        if processing == "ray":
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            tasks = [ray_evaluate_individual.remote(self, ind) for ind in individuals]
            results = ray.get(tasks)
        elif processing == "local":
            with Pool(processes=self.config.get("num_processes", 4)) as pool:
                results = pool.map(local_evaluate_individual, [(self, ind) for ind in individuals])
        else:
            results = [self.evaluate_individual(ind) for ind in individuals]
        return results

    def run(self):
        """
        Main method to run the genetic algorithm optimization.
        """
        NGEN = 100
        CXPB = 0.5
        MUTPB = 0.1
        INITIAL_POPULATION = 100

        os.makedirs(f"population/{self.session_id}", exist_ok=True)
        os.makedirs(f"weights/{self.session_id}", exist_ok=True)

        if self.gen is not None:
            pop = self.load_population(self.session_id, self.gen)
            if not pop:
                pop = self.toolbox.population(n=INITIAL_POPULATION)
        else:
            pop = self.toolbox.population(n=INITIAL_POPULATION)

        logger.info("Evaluating initial population...")
        results = self.evaluate_individuals(pop)
        for ind, (fit, avg_profit, agent) in zip(pop, results):
            ind.fitness.values = fit
            ind.avg_profit = avg_profit
            ind.agent = agent

        for gen in range(1, NGEN + 1):
            logger.info(f"=== Generation {gen} ===")
            desired_pop_size = 1000 if gen > 1 else INITIAL_POPULATION
            offspring = self.toolbox.select(pop, desired_pop_size)
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
            logger.info(f"Evaluating {len(invalid_inds)} individuals with {self.config.get('processing', 'ray')} processing...")
            results = self.evaluate_individuals(invalid_inds)
            for ind, (fit, avg_profit, agent) in zip(invalid_inds, results):
                ind.fitness.values = fit
                ind.avg_profit = avg_profit
                ind.agent = agent
            pop[:] = offspring

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

            best_ind = tools.selBest(pop, 1)[0]
            best_fitness = best_ind.fitness.values[0]
            best_reward = getattr(best_ind, 'avg_profit', None)
            best_agent = getattr(best_ind, 'agent', None)
            logger.info(f"Best Fitness: {best_fitness} | Best Reward: {best_reward}")

            for ind_id, ind in enumerate(pop):
                agent_to_save = getattr(ind, 'agent', None)
                ind_profit = getattr(ind, 'avg_profit', 0)
                if agent_to_save is not None and ind_profit > 100000:
                    use_id = str(ind_id)
                    use_profit = int(ind_profit)
                    weight_file = f"weights/{self.session_id}/gen{gen}-{use_id}-{use_profit}.pth"
                    agent_to_save.save(weight_file)
                    logger.info(f"Saved RL agent's weights to {weight_file} because profit {ind_profit} > 100000")
            for ind in pop:
                if hasattr(ind, 'agent'):
                    del ind.agent
            gc.collect()
            logger.info(f"Generation {gen} completed. Best fitness: {best_fitness} | Best reward: {best_reward}")

        logger.info("DEAP Optimization Completed")

    def load_population(self, session_id: str, generation_number: int):
        """
        Load a saved population from a JSON file.
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
        """
        pass

    def test_individual(self, individual_params=None):
        """
        Stub for testing a single individual.
        """
        pass

    def debug_single_individual(self, individual_params=None):
        """
        Debug a single individual without running the full GA loop.
        """
        from deap import creator
        if individual_params is None:
            keys = list(self.parameter_indices.keys())
            individual_params = []
            for k in keys:
                if k[0] == 'model':
                    low, high = self.model_params[k[1]]
                else:
                    indicator_name, param_name, timeframe = k
                    indicator_dict = self.indicators[indicator_name]
                    if timeframe in indicator_dict:
                        low, high = indicator_dict[timeframe][param_name]
                    else:
                        low, high = indicator_dict[param_name]
                val = random.randint(low, high) if isinstance(low, int) else random.uniform(low, high)
                individual_params.append(val)
        ind = creator.Individual(individual_params)
        logger.info("=== Debugging single individual ===")
        logger.info(f"=== Use of GPU {USING_CUDF} - Number GPUs: {NUM_GPU} ===")
        fit, avg_profit, agent = self.evaluate_individual(ind, useLocal=True)
        logger.info(f"Evaluation finished. Fit: {fit}, Profit: {avg_profit}")
        if agent is not None:
            logger.info("An RL agent was created; you can place breakpoints inside run_rl_training or environment.")
        else:
            logger.info("No RL agent was created (possibly not enough data).")
        return fit, avg_profit, agent
