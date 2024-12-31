# main.py

import argparse
import random

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Optimizer for Trading Strategy')
    parser.add_argument('--session_id', type=str, default=None, help='Session ID for this GA run')
    parser.add_argument('--gen', type=int, default=None, help='Generation number from which to load population')

    args = parser.parse_args()

    if args.session_id == None:
        session = random.randint(100000, 999999)
    else:
        session = args.session_id

    # Load data
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    # Instantiate GeneticOptimizer
    optimizer = GeneticOptimizer(
        data_loader,
        session_id=session,
        gen=args.gen
    )

    # Run the genetic optimization
    optimizer.run()


if __name__ == '__main__':
    main()
