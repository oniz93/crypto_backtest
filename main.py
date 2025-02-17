"""
main.py
-------
This is the main entry point for the genetic algorithm optimization.
It loads market data, instantiates the GeneticOptimizer, and starts the optimization process.
"""

import argparse
import random
from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer

def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Optimizer for Trading Strategy')
    parser.add_argument('--session_id', type=str, default=None, help='Session ID for this GA run')
    parser.add_argument('--gen', type=int, default=None, help='Generation number to load population from')
    args = parser.parse_args()

    session = args.session_id if args.session_id is not None else random.randint(100000, 999999)

    # Load market data.
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Instantiate the GeneticOptimizer with the loaded data.
    optimizer = GeneticOptimizer(data_loader, session_id=session, gen=args.gen)
    # Run the genetic optimization.
    optimizer.run()

if __name__ == '__main__':
    main()
