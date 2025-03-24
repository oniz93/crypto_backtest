"""
genetic_optimizer.py
--------------------
This module implements a genetic algorithm optimizer for trading strategy parameters.
It uses the DEAP library and supports distributed evaluation with Ray or multiprocessing.
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
import pandas as pd

try:
    import cudf
    import cupy as cp
    import numba.cuda as cuda
    NUM_GPU = cp.cuda.runtime.getDeviceCount()
    USING_CUDF = NUM_GPU > 0
except:
    cudf = None
    cp = None
    cuda = None
    USING_CUDF = False
    NUM_GPU = 0

from deap import base, creator, tools
import ray
from multiprocessing import Pool
from src.config_loader import Config
from src.data_loader import DataLoader
from src.dataframe_strategy import get_dataframe_strategy

import warnings
warnings.filterwarnings("ignore")

# Set up logging
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
def ray_evaluate_individual(pickled_class, individual, generation=None):
    """
    Remote function for evaluating one individual using Ray.
    """
    return pickled_class.evaluate_individual(individual, generation=generation)

def local_evaluate_individual(args):
    """
    Evaluate a single individual locally using multiprocessing.
    """
    optimizer, individual, gpuId, generation = args
    asd = optimizer.evaluate_individual(individual, gpuId=gpuId, generation=generation)
    return asd

class GeneticOptimizer:
    def __init__(self, data_loader: DataLoader, session_id=None, gen=None,
                 indicators_dir='precalculated_indicators_parquet',
                 checkpoint_dir='checkpoints', checkpoint_file=None):
        """
        Initialize the GeneticOptimizer.
        """
        self.features_df = None
        self.data_loader = data_loader
        self.df_strategy = get_dataframe_strategy()
        self.indicators_dir = indicators_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.config = Config()
        self.session_id = session_id if session_id else "unknown-session"
        self.gen = gen
        self.indicators = self.define_indicators()
        self.model_params = {'threshold_buy': (0.5, 0.9), 'threshold_sell': (0.5, 0.9)}
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
            # 'vwma': {'length': (5, 200)},
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
            # 'stoch': {'k': (5, 20), 'd': (3, 10)},
            'atr': {'length': (5, 30)},
            'adx': {'length': (5, 200)},
            # 'roc': {'length': (5, 30)},
            # 'hist_vol': {'length': (5, 50)},
            # 'obv': {}
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
                            raise ValueError(f"Indicator '{indicator_name}' timeframe '{tf}' param '{param_name}' has invalid limits: {limits}")
                else:
                    # Non-timeframe-specific parameters (e.g., 'vwap', 'rsi')
                    for param_name, limits in data.items():
                        try:
                            low, high = limits
                            parameter_indices[(indicator_name, param_name, tf)] = idx
                            idx += 1
                        except ValueError:
                            raise ValueError(f"Indicator '{indicator_name}' param '{param_name}' has invalid limits: {limits}")
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

    def evaluate_individual(self, individual, useLocal=False, gpuId=None, generation=None):
        """
        Evaluate an individual by computing indicator features, filtering data,
        and running RL training.
        """
        import gc
        import torch
        if gpuId != None:
            cuda.select_device(gpuId)
        config = self.extract_config_from_individual(individual)
        if useLocal:
            hash_str = hashlib.md5(str(individual).encode()).hexdigest()
            features_file = f"features/{hash_str}.parquet"
            if os.path.exists(features_file):
                logger.info("Loading precomputed indicator features from file...")
                self.features_df = self.df_strategy.load_parquet(features_file)  # Always pandas for saved files
            else:
                logger.info("Computing indicator features for individual (local mode)...")
                self.load_and_prepare_features(config)
                self.features_df.to_parquet(features_file)
        else:
            logger.info("Computing indicator features for individual...")
            self.load_and_prepare_features(config)

        self.features_df = self.data_loader.filter_data_by_date(
            self.features_df,
            self.config.get('start_simulation'),
            self.config.get('end_simulation')
        )

        if len(self.features_df) < 100:
            logger.warning("Not enough data to run RL.")
            if USING_CUDF: # CRUCIAL: Force cleanup.
                del self.features_df
                gc.collect()
                cuda.current_context().deallocations.clear() # Free all the CUDF
            return (9999999.0,), 0.0, None

        if 'close' not in self.features_df.columns:
            logger.error("'close' column missing in features_df.")
            if USING_CUDF: # CRUCIAL: Force cleanup.
                del self.features_df
                gc.collect()
                cuda.current_context().deallocations.clear() # Free all the CUDF
            return (9999999.0,), 0.0, None

        price_data = self.features_df[['close']]
        indicators_only = self.features_df.drop(columns=['close'], errors='ignore')
        del self.features_df
        gc.collect()
        env = self.create_environment(price_data, indicators_only)
        del indicators_only
        del price_data
        gc.collect()
        logger.info("Starting RL training for individual...")
        agent, avg_profit = self.run_rl_training(env, episodes=self.config.get('episodes', 20), generation=generation)
        del env

        if USING_CUDF:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

        gc.collect()
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
        data_loader, indicator_name, timeframe, params = args
        action_start = time.time()
        # logger.info(f"_calculate_indicator_worker - Starting calculation for {indicator_name} on TF {timeframe} - Using cuDF: {USING_CUDF}")
        try:
            indicator_df = data_loader.calculate_indicator(indicator_name, params, timeframe)
            indicator_df = indicator_df.add_suffix(f'_{indicator_name}_{timeframe}')
            # logger.info(f"Calculation done {indicator_name} on TF {timeframe} took {time.time() - action_start:.2f}s")
            if timeframe != data_loader.base_timeframe:
                shift_duration = pd.to_timedelta(timeframe)
                indicator_df_shifted = indicator_df.copy()
                indicator_df_shifted.index = indicator_df_shifted.index - shift_duration
                indicator_df = indicator_df_shifted.resample(data_loader.base_timeframe).ffill()
                indicator_df.dropna(inplace=True)
            return (indicator_name, timeframe, indicator_df)
        except Exception as e:
            logger.error(f"Error in _calculate_indicator_worker for {indicator_name} on TF {timeframe}: {e}")
            raise

    def load_and_prepare_features(self, config):
        """
        Compute and merge technical indicators into a feature DataFrame in parallel.
        Merges each indicator DataFrame as soon as it's computed, leveraging GPU acceleration with cuDF.

        Parameters:
            config (dict): Configuration containing indicator_params, num_processes, and simulation dates

        Returns:
            DataFrame: Final feature DataFrame (cuDF or pandas) with all indicators and additional features
        """
        # Extract configuration
        indicator_params = config['indicator_params']

        # Prepare tasks
        tasks = [(self.data_loader, indicator_name, timeframe, params)
                 for indicator_name, timeframes_params in indicator_params.items()
                 for timeframe, params in timeframes_params.items()]

        # Initialize features_df
        self.features_df = (self.base_price_data.copy() if isinstance(self.base_price_data, cudf.DataFrame)
                                else cudf.from_pandas(self.base_price_data) if USING_CUDF
            else self.base_price_data.copy())

        # logger.info(f"Initialized features_df as {type(self.features_df)}")

        for task in tasks:
            indicator_name, timeframe, indicator_df = self._calculate_indicator_worker(task)
            # logger.info(f"Received result for {indicator_name} on TF {timeframe}")

            indicator_df.columns = [col.replace('\n', '').strip() for col in indicator_df.columns]
            self.features_df = self.features_df.join(indicator_df)
            del indicator_df
            gc.collect()

        # Compute RSI Divergence (example, adjust column names as needed)
        # Assuming indicators append timeframe to column names, e.g., 'rsi_1min', 'rsi_5min'
        try:
            rsi_base_col = next(col for col in self.features_df.columns if
                                'rsi' in col.lower() and self.data_loader.base_timeframe in col)
            rsi_5min_col = next(col for col in self.features_df.columns if 'rsi' in col.lower() and '5T' in col)
            self.features_df['RSI_Divergence'] = self.features_df[rsi_base_col] - self.features_df[rsi_5min_col]
        except StopIteration:
            logger.warning("RSI columns not found; skipping RSI_Divergence calculation")

        # Drop NaN values
        self.features_df = self.features_df.dropna()

        # Exit early for debugging - even before processing all indicators
        # logger.info("Exiting early for debugging")
        # exit()
        
        # Filter by date (not reached due to exit)
        self.features_df = self.data_loader.filter_data_by_date(
            self.features_df, self.config.get('start_simulation'), self.config.get('end_simulation')
        )
        # logger.info(f"Final features_df type: {type(self.features_df)}")
        return True

    def create_environment(self, price_data, indicators):
        """
        Create a TradingEnvironment instance using price and indicator data.
        """
        from src.rl_environment import TradingEnvironment
        initial_capital = 100000
        transaction_cost = 0.005
        mode = self.config.get('training_mode')
        return TradingEnvironment(price_data, indicators, initial_capital=initial_capital,
                                 transaction_cost=transaction_cost, mode=mode, using_gpu=USING_CUDF)

    def run_rl_training(self, env, episodes=1, generation=None):
        """
        Run reinforcement learning training on the trading environment using dataset-level chunking.
        """
        from src.rl_agent import DQNAgent
        import gc
        import torch

        seq_length = self.config.get('seq_length', 720)
        chunk_size = self.config.get('chunk_size', 4000)
        batch_frequency = 32

        agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-3, seq_length=seq_length, epsilon_decay=0.995, use_cudf=USING_CUDF)
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
                if generation is not None and i >= generation:
                    break
                chunk_len = len(indices)
                logger.info(f"Episode {ep + 1}: Processing chunk {i + 1}/{num_chunks} (size {chunk_len} rows) | Max chunks: {generation}")
                env.data_values = full_data_np[indices]
                env.n_steps = len(indices)
                if USING_CUDF:
                    env.timestamps_list = env.data.index[indices].to_pandas().tolist()
                else:
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
                    penalty_factor = (chunk_len - step_count) / chunk_len
                    chunk_reward *= (1 + penalty_factor)
                ep_reward += chunk_reward
                logger.info(f"Chunk {i + 1} took {time.time() - chunk_start_time:.2f}s reward: {chunk_reward:.2f} | N+sell {profit_sells} -> {profit_total:.2f} | N-sell {loss_sells} -> {loss_total:.2f}")
            total_reward_sum += ep_reward
            episode_count += 1
            logger.info(f"Episode {ep + 1} completed. Total episode reward: {ep_reward:.2f}")

        logger.info(f"RL training over {episodes} episodes completed. Reward: {total_reward_sum:.2f}")
        del full_data_np
        del env.data_values
        del env.data
        del agent.env

        if USING_CUDF:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

        gc.collect()
        return agent, total_reward_sum

    def evaluate_individuals(self, individuals, generation=None):
        processing = self.config.get("processing", "ray").lower()
        if processing == "ray":
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            tasks = [ray_evaluate_individual.remote(self, ind, generation) for ind in individuals]
            results = ray.get(tasks)
        elif processing == "local":
            if USING_CUDF:
                num_processes = NUM_GPU
                pools = []
                results = []
                individual_chunks = [[] for _ in range(num_processes)]  # List of lists to hold individuals for each GPU

                # Distribute individuals to GPU chunks in a round-robin fashion
                gpu_id_counter = 0
                for ind in individuals:
                    gpu_id = gpu_id_counter % num_processes
                    individual_chunks[gpu_id].append(ind)
                    gpu_id_counter += 1

                # Create a pool for each GPU and process assigned individuals
                for gpu_id in range(num_processes):
                    pool = Pool(processes=1)  # Each pool has only 1 process, dedicated to a GPU
                    pools.append(pool)
                    args_list_for_gpu = [(self, ind, gpu_id, generation) for ind in
                                         individual_chunks[gpu_id]]  # Prepare args for this GPU
                    async_result = pool.map_async(local_evaluate_individual,
                                                  args_list_for_gpu)  # Use map_async for non-blocking execution
                    results.append(async_result)

                # Collect results from all pools - this will block until all pools are done
                final_results = []
                for async_result in results:
                    final_results.extend(async_result.get())  # Get results from each pool and extend the final list
                results = final_results

                for pool in pools:  # Close all pools
                    pool.close()
                    pool.join()

            else:  # If not using CUDF, fallback to the original local processing
                num_processes = self.config.get("num_processes", 4)
                with Pool(processes=num_processes) as pool:
                    results = pool.map(local_evaluate_individual, [(self, ind, None, generation) for ind in individuals])
        else:
            results = [self.evaluate_individual(ind, generation=generation) for ind in individuals]
        return results

    def run(self):
        """
        Main method to run the genetic algorithm optimization.
        """
        NGEN = 100
        CXPB = 0.5
        MUTPB = 0.1
        INITIAL_POPULATION = 1000

        os.makedirs(f"population/{self.session_id}", exist_ok=True)
        os.makedirs(f"weights/{self.session_id}", exist_ok=True)

        if self.gen is not None:
            pop = self.load_population(self.session_id, self.gen)
            if not pop:
                pop = self.toolbox.population(n=INITIAL_POPULATION)
        else:
            pop = self.toolbox.population(n=INITIAL_POPULATION)

        logger.info("Evaluating initial population...")
        # results = self.evaluate_individuals(pop, generation=1)
        # for ind, (fit, avg_profit, agent) in zip(pop, results):
        #     ind.fitness.values = fit
        #     ind.avg_profit = avg_profit
        #     ind.agent = agent

        for gen in range(1, NGEN + 1):
            logger.info(f"=== Generation {gen} ===")
            desired_pop_size = 100 if gen > 1 else INITIAL_POPULATION
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
            results = self.evaluate_individuals(invalid_inds, generation=gen)
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
        fit, avg_profit, agent = self.evaluate_individual(ind, useLocal=True, gpuId=1)
        logger.info(f"Evaluation finished. Fit: {fit}, Profit: {avg_profit}")
        if agent is not None:
            logger.info("An RL agent was created; you can place breakpoints inside run_rl_training or environment.")
        else:
            logger.info("No RL agent was created (possibly not enough data).")
        return fit, avg_profit, agent
