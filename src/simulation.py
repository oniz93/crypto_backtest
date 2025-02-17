"""
simulation.py
-------------
This module defines a simple Simulation class that runs a trading strategy.
It serves as a high-level interface to run a simulation (e.g. training or evaluation)
by calling the strategy’s calculate_profit() method.
"""

class Simulation:
    def __init__(self, strategy):
        """
        Initialize the Simulation with a given trading strategy.

        Parameters:
            strategy: An object that implements a calculate_profit() method.
                      This strategy might be based on a reinforcement learning agent,
                      a genetic algorithm optimizer, or any other approach.
        """
        # Save the provided strategy so that it can be used later to run the simulation.
        self.strategy = strategy

    def run(self):
        """
        Runs the simulation by invoking the strategy’s calculate_profit() method.

        Returns:
            The result of the simulation (for example, the computed profit).
        """
        # Call the calculate_profit method from the strategy
        result = self.strategy.calculate_profit()
        # Return the simulation result to the caller.
        return result
