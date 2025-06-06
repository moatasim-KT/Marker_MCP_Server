"""Configuration settings for the Marker MCP Server.

This module handles configuration management including loading from environment variables,
configuration files, and providing default values.
"""
import os
import logging
from typing import Dict, Any
from pydantic import ValidationError
from .config_schema import AppConfig

from .utils import setup_logging

# Default configuration
DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 8000,
        "log_level": "INFO",
        "debug": False,
    },
    "marker": {
        "device": None,  # Auto-detect
        "batch_size": 1,
        "max_pages": None,  # No limit
        "parallel_factor": 1,
    },
    "paths": {
        "cache_dir": os.path.expanduser("~/.cache/marker-mcp"),
        "model_dir": os.path.expanduser("~/.cache/marker-mcp/models"),
    },
}


class Config:
    """Configuration manager for the Marker MCP Server."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config = AppConfig()
            self._load_from_env()
            self._initialized = True
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Server settings
        if host := os.getenv("MARKER_MCP_HOST"):
            self._config.server.host = host
        if port := os.getenv("MARKER_MCP_PORT"):
            try:
                self._config.server.port = int(port)
            except (ValueError, TypeError):
                pass
        if log_level := os.getenv("MARKER_MCP_LOG_LEVEL"):
            self._config.server.log_level = log_level.upper()
        
        # Marker settings
        if device := os.getenv("MARKER_DEVICE"):
            self._config.marker.device = device
        if batch_size := os.getenv("MARKER_BATCH_SIZE"):
            try:
                self._config.marker.batch_size = int(batch_size)
            except (ValueError, TypeError):
                pass
        
        # Paths
        if cache_dir := os.getenv("MARKER_CACHE_DIR"):
            self._config.paths.cache_dir = cache_dir
        if model_dir := os.getenv("MARKER_MODEL_DIR"):
            self._config.paths.model_dir = model_dir
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'server.host')
            default: Default value if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except (KeyError, TypeError, AttributeError):
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates using dot notation keys
        """
        for key, value in updates.items():
            keys = key.split('.')
            d = self._config
            for k in keys[:-1]:
                d = getattr(d, k)
            setattr(d, keys[-1], value)
    
    def setup_logging(self) -> None:
        """Set up logging based on the configuration."""
        log_level = self.get("server.log_level", "INFO")
        log_level_map = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        level = log_level_map.get(log_level.upper(), 20)  # Default to INFO
        
        setup_logging(level=level)
    
    def get_logger(self, name: str) -> 'logging.Logger':
        """Get a logger with the specified name.
        
        Args:
            name: The name of the logger
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Ensure the logger has at least one handler
        if not logger.handlers:
            # Create a console handler if none exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Set the log level from config if not set
            if logger.level == logging.NOTSET:
                log_level = self.get("server.log_level", "INFO")
                log_level_map = {
                    "DEBUG": logging.DEBUG,
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL,
                }
                logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))
        
        return logger


# Global configuration instance
try:
    config = Config()
except ValidationError as e:
    print(f"Configuration validation error: {e}")
    raise
