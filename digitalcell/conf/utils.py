import yaml
from pathlib import Path
from typing import Dict

def recursive_update(base_dict: Dict, override_dict:Dict) -> None:
    """
    Recursively updates a base dictionary with values from an override dictionary.
    If a key in override_dict is also a dictionary, it merges the sub-dictionaries.
    """
    for key, value in override_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            # Recursively merge dictionaries
            recursive_update(base_dict[key], value)
        else:
            # Otherwise, replace the value
            base_dict[key] = value


def load_config(experiment_file: Path) -> Dict:
    # Load the override configuration file first
    with open(experiment_file, 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    # Ensure the override config contains the path to the default config
    if 'default_config' not in experiment_config:
        return experiment_config
    
    default_file = experiment_config.pop('default_config')
    
    # Load the default configuration
    with open(default_file, 'r') as f:
        default_config = yaml.safe_load(f)
    
    # Update the default config with the override values
    recursive_update(default_config, experiment_config)
    return default_config
