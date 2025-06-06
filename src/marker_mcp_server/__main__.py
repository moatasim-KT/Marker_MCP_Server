# Copyright (c) 2024 Marker MCP Server Authors
"""Marker MCP Server entrypoint module.

This is the entry point for the Marker MCP Server.
"""
import asyncio
import logging
import sys
from typing import Optional, List

from . import __version__
from .server import main as server_main
from .utils import get_logger

logger = get_logger(__name__)


def parse_args(args: Optional[List[str]] = None) -> dict:
    """Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        dict: Parsed arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Marker MCP Server - PDF conversion tools via MCP protocol"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return vars(parser.parse_args(args))


async def async_main() -> None:
    """Run the Marker MCP server asynchronously."""
    try:
        logger.info(f"Starting Marker MCP Server v{__version__}")
        await server_main()
    except asyncio.CancelledError:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Marker MCP Server stopped")


def run() -> None:
    """Run the Marker MCP server."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.get("verbose") else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # Run the async main function
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
