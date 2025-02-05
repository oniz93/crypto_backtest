# debug_example.py

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer

def debug_single():
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    params = [
        0, 0, 0, 0, 0, 0, 0, # VWAP
        50000, 10000, 3500, 2000, 1000, 250, 60, # VWMA
        50000, 10000, 3500, 2000, 1000, 250, 60, # VPVR
        14, 14, 14, 14, 14, 14, 14, # RSI
        8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, # MACD
        0.0, 0.0
    ]
    go = GeneticOptimizer(data_loader, session_id="debug123")
    go.debug_single_individual(params)

if __name__ == "__main__":
    debug_single()
