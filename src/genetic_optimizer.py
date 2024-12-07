# src/genetic_optimizer.py

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import multiprocessing

class GeneticOptimizer:
    def __init__(self, strategy_class, data_loader):
        self.strategy_class = strategy_class
        self.data_loader = data_loader
        self.indicator_cache = {}
        self.indicators = self.define_indicators()
        self.timeframes = [
            # '1T', '5T', '15T', '30T', '45T', '1H', '2H', '4H', '8H', '12H', '1D'
            '1T', '5T', '15T', '30T',  '1H', '4H', '1D'
        ]
        # Define model parameters and their ranges
        self.model_params = {
            'threshold_buy': (0.5, 0.9),
            'threshold_sell': (0.5, 0.9),
            'model_C': (0.01, 10)
        }
        self.parameter_indices = self.create_parameter_indices()
        # Prepare data once for efficiency
        self.prepare_data()

    def define_indicators(self):
        """
        Define the indicators and their parameter ranges.
        """
        return {
            'sma': {'length': (5, 200)},
            'ema': {'length': (5, 200)},
            'rsi': {'length': (5, 30)},
            'macd': {'fast': (5, 20), 'slow': (21, 50), 'signal': (5, 20)},
            'bbands': {'length': (5, 50), 'std_dev': (1.0, 3.0)},
            'atr': {'length': (5, 50)},
            'stoch': {'k': (5, 20), 'd': (3, 10)},
            # 'cci': {'length': (5, 50)},
            'adx': {'length': (5, 50)},
            # 'cmf': {'length': (5, 50)},
            # 'mfi': {'length': (5, 30)},
            # 'roc': {'length': (5, 50)},
            # 'willr': {'length': (5, 30)},
            # 'psar': {'acceleration': (0.01, 0.1), 'max_acceleration': (0.1, 0.5)},
            'ichimoku': {'tenkan': (5, 20), 'kijun': (20, 60), 'senkou': (40, 100)},
            # 'keltner': {'length': (5, 50), 'multiplier': (1.0, 3.0)},
            # 'donchian': {'lower_length': (5, 50), 'upper_length': (5, 50)},
            # 'emv': {'length': (5, 20)},
            # 'force': {'length': (1, 20)},
            # 'uo': {'short': (5, 7), 'medium': (8, 14), 'long': (15, 28)},
            # 'volatility': {'length': (5, 50)},
            # 'dpo': {'length': (5, 50)},
            # 'trix': {'length': (5, 50)},
            # 'chaikin_osc': {'fast': (3, 10), 'slow': (10, 20)},
            'vwap': {},
            # Indicators without parameters are not included
        }

    def create_parameter_indices(self):
        """
        Create a mapping of parameter indices for the individual's genes.
        """
        parameter_indices = {}
        idx = 0
        # Add indicator parameters
        for indicator_name, params in self.indicators.items():
            for param_name in params.keys():
                for timeframe in self.timeframes:
                    key = ('indicator', indicator_name, param_name, timeframe)
                    parameter_indices[key] = idx
                    idx += 1
        # Add model parameters
        for param_name in self.model_params.keys():
            key = ('model', param_name)
            parameter_indices[key] = idx
            idx += 1
        return parameter_indices

    def get_total_parameters(self):
        """
        Get the total number of parameters.
        """
        return len(self.parameter_indices)

    def get_varbound_and_vartype(self):
        """
        Create varbound and vartype arrays for the genetic algorithm.
        """
        total_params = self.get_total_parameters()
        varbound = np.zeros((total_params, 2))
        vartype = []
        for key, idx in self.parameter_indices.items():
            if key[0] == 'indicator':
                _, indicator_name, param_name, timeframe = key
                param_range = self.indicators[indicator_name][param_name]
            elif key[0] == 'model':
                _, param_name = key
                param_range = self.model_params[param_name]
            else:
                continue  # Skip if key doesn't match expected format
            varbound[idx, 0] = param_range[0]
            varbound[idx, 1] = param_range[1]
            if isinstance(param_range[0], int):
                vartype.append('int')
            else:
                vartype.append('real')
        vartype = np.array(vartype).reshape(-1, 1)
        return varbound, vartype

    def extract_config_from_individual(self, individual):
        """
        Extract the configuration from the individual's genes.
        """
        config = {}
        indicator_params = {}
        model_params = {}
        for key, idx in self.parameter_indices.items():
            value = individual[idx]
            if key[0] == 'indicator':
                _, indicator_name, param_name, timeframe = key
                if indicator_name not in indicator_params:
                    indicator_params[indicator_name] = {}
                if timeframe not in indicator_params[indicator_name]:
                    indicator_params[indicator_name][timeframe] = {}
                indicator_params[indicator_name][timeframe][param_name] = value
            elif key[0] == 'model':
                _, param_name = key
                model_params[param_name] = value
        config['indicator_params'] = indicator_params
        config['model_params'] = model_params
        # Add other global parameters if needed
        return config

    def load_indicators(self, config):
        """
        Load indicators based on the individual's parameters.
        """
        indicators = {}
        indicator_params = config['indicator_params']
        print("Total indicators to load:", len(indicator_params))
        for indicator_name, timeframes_params in indicator_params.items():
            indicators[indicator_name] = {}
            for timeframe, params in timeframes_params.items():
                print(f"Loading {indicator_name} for {timeframe}")
                key = (indicator_name, tuple(params.items()), timeframe)
                if key in self.indicator_cache:
                    indicator_df = self.indicator_cache[key]
                else:
                    indicator_df = self.data_loader.calculate_indicator(
                        indicator_name, params, timeframe)
                    self.indicator_cache[key] = indicator_df
                indicators[indicator_name][timeframe] = indicator_df
        return indicators

    def prepare_data(self):
        """
        Prepares the data required for training models.
        """
        base_tf = self.data_loader.base_timeframe
        price_data = self.data_loader.tick_data[base_tf].copy()

        # Create labels for buy and sell signals
        price_data['future_return'] = price_data['close'].shift(-1) / price_data['close'] - 1
        price_data['buy_signal'] = np.where(price_data['future_return'] > 0, 1, 0)
        price_data['sell_signal'] = np.where(price_data['future_return'] < 0, 1, 0)

        # Store labels
        self.labels_buy = price_data['buy_signal']
        self.labels_sell = price_data['sell_signal']

        # Store base price data
        self.base_price_data = price_data

    def prepare_features(self, indicators):
        """
        Prepares the features DataFrame from the indicators.
        """
        # Start with base price data
        features_df = self.base_price_data.copy()

        # Merge the indicators into one DataFrame
        for indicator_name, tf_dict in indicators.items():
            for tf, df in tf_dict.items():
                df = df.reindex(features_df.index, method='bfill').add_suffix(f'_{indicator_name}_{tf}')
                features_df = features_df.join(df)

        # Drop rows with NaN values
        features_df.dropna(inplace=True)

        # Exclude price columns and other non-feature columns
        features = features_df.drop(columns=[
            'open', 'high', 'low', 'close', 'volume',
            'future_return', 'buy_signal', 'sell_signal'
        ])

        # Align labels with features
        self.labels_buy = self.labels_buy.loc[features.index]
        self.labels_sell = self.labels_sell.loc[features.index]

        return features

    def train_models(self, X_train, y_train_buy, y_train_sell, model_params):
        """
        Trains the models using the features and labels.
        """
        model_C = model_params['model_C']
        from sklearn.linear_model import LogisticRegression

        model_buy = LogisticRegression(C=model_C, max_iter=1000)
        model_buy.fit(X_train, y_train_buy)

        model_sell = LogisticRegression(C=model_C, max_iter=1000)
        model_sell.fit(X_train, y_train_sell)

        return model_buy, model_sell

    def evaluate(self, individual):
        """
        Evaluate the fitness of an individual.
        """
        config = self.extract_config_from_individual(individual)
        indicator_params = config['indicator_params']
        model_params = config['model_params']

        # Load indicators based on the individual's parameters
        indicators = self.load_indicators(config)
        # Update data_loader indicators
        self.data_loader.indicators = indicators

        # Prepare features and labels
        features = self.prepare_features(indicators)
        labels_buy = self.labels_buy
        labels_sell = self.labels_sell

        # Split features and labels into training and testing sets
        split_index = int(len(features) * 0.8)
        X_train = features.iloc[:split_index]
        y_train_buy = labels_buy.iloc[:split_index]
        y_train_sell = labels_sell.iloc[:split_index]
        # X_test and y_test can be used for evaluation if needed

        # Train models using model parameters
        model_buy, model_sell = self.train_models(X_train, y_train_buy, y_train_sell, model_params)

        # Create configuration for the strategy
        strategy_config = {
            'threshold_buy': model_params['threshold_buy'],
            'threshold_sell': model_params['threshold_sell'],
            # Include other config parameters if needed
        }

        # Initialize the TradingStrategy with the trained models and config
        strategy = self.strategy_class(self.data_loader, strategy_config, model_buy, model_sell)

        # Run the strategy to get profit
        profit = strategy.calculate_profit()

        return -profit  # Negative profit for minimization

    def run(self):
        varbound, vartype = self.get_varbound_and_vartype()
        algorithm_param = {
            'max_num_iteration': None,
            'population_size': 50,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'two_point',
            'max_iteration_without_improv': None,
            'multiprocessing_ncpus': multiprocessing.cpu_count(),
            'multiprocessing_engine': None
        }
        model = ga(
            function=self.evaluate,
            dimension=self.get_total_parameters(),
            variable_type_mixed=vartype,
            variable_boundaries=varbound,
            function_timeout=1000000,
            algorithm_parameters=algorithm_param
        )
        model.run()

    def test_individual(self, individual_params=None):
        """
        Creates a single individual and runs evaluate() for testing purposes.

        Parameters:
        - individual_params: list or numpy array of parameter values. If None, a random individual is created.

        Returns:
        - The fitness value of the individual.
        """
        if individual_params is None:
            # Create a random individual within parameter bounds
            total_params = self.get_total_parameters()
            varbound, vartypes = self.get_varbound_and_vartype()
            individual = []
            for idx in range(total_params):
                low, high = varbound[idx]
                vartype = vartypes[idx][0]
                if vartype == 'int':
                    value = np.random.randint(low, high + 1)
                else:
                    value = np.random.uniform(low, high)
                individual.append(value)
        else:
            # Use the provided individual parameters
            individual = individual_params

        # Evaluate the individual
        fitness = self.evaluate(individual)
        print(f"Fitness of the test individual: {fitness}")

        return fitness

    def print_parameter_indices(self):
        print("Parameter Indices:")
        for key, idx in sorted(self.parameter_indices.items(), key=lambda x: x[1]):
            print(f"Index {idx}: {key}")