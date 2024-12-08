# src/config_loader.py

import yaml
import os

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, section: str, key: str = None):
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section, {})

    def __getattr__(self, item):
        return self.config.get(item, None)