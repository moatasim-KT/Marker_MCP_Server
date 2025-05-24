import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'optimized_config.json'


def load_config() -> dict:
    """Load configuration from JSON file."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def save_config(config: dict):
    """Save configuration to JSON file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
