# src/config_loader.py

import os

import yaml


class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            # self.config should now be a dictionary

    def get(self, key, default=None):
        return self.config.get(key, default)
