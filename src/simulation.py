# src/simulation.py

class Simulation:
    def __init__(self, strategy):
        self.strategy = strategy

    def run(self):
        result = self.strategy.calculate_profit()
        return result
