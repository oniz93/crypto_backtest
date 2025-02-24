# debug_example.py

from src.data_loader import DataLoader
from src.genetic_optimizer import GeneticOptimizer

def debug_single():
    data_loader = DataLoader()
    data_loader.import_ticks()
    data_loader.resample_data()

    params = [
        0, 2, 4, 8, 16, 8, 4, # VWAP
        200, 200, 200, 200, 200, 200, 200, # VWMA
        10, 20, 30, 40, 50, 100, 200, # VPVR
        8, 10, 12, 14, 16, 18, 20, # RSI
        8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, 8, 21, 5, # MACD
        20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0, 20, 2.0,  # BBands length, std_dev (7 timeframes)
        14, 3, 14, 3, 14, 3, 14, 3, 14, 3, 14, 3, 14, 3,  # Stoch k, d (7 timeframes)
        14, 14, 14, 14, 14, 14, 14,  # ROC length (7 timeframes)
        20, 20, 20, 20, 20, 20, 20,  # Hist_Vol length (7 timeframes)
        14, 14, 14, 14, 14, 14, 14,  # ATR length (7 timeframes)
        0.6, 0.6  # Model thresholds (buy, sell)
    ]
    go = GeneticOptimizer(data_loader, session_id="debug123")
    go.debug_single_individual(params)

if __name__ == "__main__":
    debug_single()
