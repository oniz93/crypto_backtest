# main.py

from src.data_loader import DataLoader
from src.trading_strategy import TradingStrategy
from src.simulation import Simulation
from src.genetic_optimizer import GeneticOptimizer


def main():
    # Load data
    data_loader = DataLoader()
    data_loader.import_ticks()

    # Set up initial configuration
    config = {
        'initial_balance': 100000000,
        'short_long': 'l',
        # Other configuration parameters can be set here
    }

    # Create trading strategy instance
    strategy = TradingStrategy(data_loader, config)

    # Run simulation
    simulation = Simulation(strategy)
    result = simulation.run()
    print(f"Simulation result: {result}")

    # Run genetic optimization
    optimizer = GeneticOptimizer(strategy, data_loader)
    optimizer.run()


if __name__ == '__main__':
    main()
