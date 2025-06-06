"""Marker MCP Server package.

This package provides an MCP server for interacting with the Marker PDF conversion tools.
"""

__version__ = "0.3.0"

# Import key components to make them available at the package level
from .server import main
from .tools import (
    handle_batch_convert,
    handle_single_convert,
    handle_chunk_convert,
    handle_start_server
)
from .utils import (
    get_logger,
    MarkerError,
    ResourceError
)

# Initialize package logger
logger = get_logger(__name__)

# Define public API
__all__ = [
    # Core components
    'main',
    'get_logger',
    
    # Tool handlers
    'handle_batch_convert',
    'handle_single_convert',
    'handle_chunk_convert',
    'handle_start_server',
    
    # Exceptions
    'MarkerError',
    'ResourceError',
    
    # Version
    '__version__',
]
