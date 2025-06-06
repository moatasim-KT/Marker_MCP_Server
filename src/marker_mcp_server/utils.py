"""Utility functions for the Marker MCP Server.

This module contains shared utility functions used across the package.
"""
import asyncio
import logging
import os
import sys
import traceback
from contextlib import contextmanager
from io import StringIO
from typing import Any, Callable, Generator, List, Tuple, TypeVar, cast

from mcp import types

# Type variable for generic function type
F = TypeVar('F', bound=Callable[..., Any])

def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with the given name.
    
    Args:
        name: Logger name (defaults to caller's module name if None)
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if name is None:
        # Get the name of the calling module
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'marker_mcp_server')
        else:
            name = 'marker_mcp_server'
    
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only add handlers if they haven't been added
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = get_logger()


class MarkerError(Exception):
    """Base exception for all Marker MCP Server errors."""
    pass


class ToolExecutionError(MarkerError):
    """Raised when a tool execution fails."""
    pass


class ResourceError(MarkerError):
    """Raised when there's an error with resources (e.g., models, devices)."""
    pass


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    # Set log level for third-party libraries
    logging.getLogger("marker").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def capture_output() -> Generator[Tuple[StringIO, StringIO], None, None]:
    """Context manager to capture stdout and stderr.
    
    Yields:
        tuple: (stdout_io, stderr_io) - StringIO objects containing captured output
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_io, stderr_io = StringIO(), StringIO()
    sys.stdout, sys.stderr = stdout_io, stderr_io
    try:
        yield stdout_io, stderr_io
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def run_async(func: F) -> F:
    """Decorator to run a synchronous function in a thread pool.
    
    Args:
        func: The synchronous function to wrap
        
    Returns:
        The async wrapper function
    """
    import functools
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )
    
    return cast(F, wrapper)


def get_absolute_path(path: str) -> str:
    """Convert a relative path to an absolute path.
    
    Args:
        path: The path to convert (can be relative or absolute)
        
    Returns:
        str: The absolute path
    """
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.getcwd(), path))


def create_text_content(text: str) -> List[types.TextContent]:
    """Helper to create a list with a single TextContent item.
    
    Args:
        text: The text content
        
    Returns:
        List[types.TextContent]: A list containing a single TextContent item
    """
    return [types.TextContent(type="text", text=text)]


def create_error_content(error: Exception) -> List[types.TextContent]:
    """Create an error message from an exception.
    
    Args:
        error: The exception to convert to an error message
        
    Returns:
        List[types.TextContent]: A list containing an error message
    """
    error_msg = f"Error: {str(error)}\n{traceback.format_exc()}"
    return create_text_content(error_msg)


def validate_directory(path: str) -> bool:
    """Validate that a directory exists and is accessible.
    
    Args:
        path: Path to the directory
        
    Returns:
        bool: True if the directory is valid, False otherwise
    """
    try:
        return os.path.isdir(path) and os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except (OSError, TypeError):
        return False
