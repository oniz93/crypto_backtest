"""
config_loader.py
----------------
This module defines a Config class that loads configuration settings
from a YAML file (by default 'config.yaml'). This configuration is used
across the project (e.g., for setting training parameters, date ranges, etc.).
"""

import os
import yaml

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the configuration loader.

        Parameters:
            config_path (str): Path to the YAML configuration file.
                               Defaults to 'config.yaml'.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        # Check if the configuration file exists.
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        # Open the YAML file and load it as a dictionary.
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)  # Now self.config is a dictionary

    def get(self, key, default=None):
        """
        Get a configuration value.

        Parameters:
            key: The configuration key.
            default: The default value to return if the key is not present.

        Returns:
            The configuration value or the default if key is missing.
        """
        # Return the value for key if it exists; otherwise, return the default.
        return self.config.get(key, default)
