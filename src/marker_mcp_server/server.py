# Copyright (c) 2024 Marker MCP Server Authors
"""Marker MCP Server.

This module implements an MCP server for interacting with Marker conversion tools.
"""
import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
import traceback
import uvicorn
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import FastMCP

# Import dashboard
from .dashboard import app as dashboard_app

from .tools import (
    handle_batch_convert,
    handle_batch_pages_convert,
    handle_chunk_convert,
    handle_single_convert,
    handle_start_server,
)
from .monitoring import initialize_monitoring, get_metrics_collector, shutdown_monitoring
from .config import Config

if TYPE_CHECKING:
    from logging import Logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("marker-mcp-server")

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.
    
    Args:
        host: The host to check
        port: The port to check
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except (socket.error, OSError):
        return False


def find_available_port(start_port: int, host: str = "127.0.0.1", max_attempts: int = 10) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: The starting port to check
        host: The host to bind to
        max_attempts: Maximum number of ports to check
        
    Returns:
        An available port number, or -1 if none found
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return -1


def safe_int(val: object) -> int:
    """Safely convert a value to int, returning 0 if conversion fails.

    or type is unsupported.

    Args:
        val: The value to convert.

    Returns:
        The integer value, or 0 if conversion is not possible.

    """
    if isinstance(val, (int, float, str)):
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0
    return 0


async def handle_tool_call(
    name: str, arguments: dict[str, Any],
) -> dict[str, Any] | str:
    """Handle a tool call by name with arguments.

    Args:
        name: The name of the tool to call.
        arguments: The arguments to pass to the tool.

    Returns:
        The result of the tool call. For most tools, this will be a dictionary with
        success/error information.

    """
    logger.info(
        json.dumps(
            {"event": "tool_call", "tool": name, "arguments": arguments},
            default=str,
        ),
    )
    
    # Debug: Write arguments to file for inspection
    try:
        with open("debug_handle_tool_call.txt", "w") as f:
            f.write(f"handle_tool_call called with name: {name}\n")
            f.write(f"handle_tool_call arguments: {arguments}\n")
            f.write(f"arguments type: {type(arguments)}\n")
            f.write(f"arguments keys: {list(arguments.keys()) if isinstance(arguments, dict) else 'not a dict'}\n")
    except Exception as e:
        logger.error(f"Failed to write debug file: {e}")
    
    # Get metrics collector and start tracking
    metrics_collector = get_metrics_collector()
    job_id = None
    
    try:
        # Start operation tracking
        if metrics_collector:
            file_path = arguments.get("file_path") or arguments.get("fpath")
            job_id = metrics_collector.start_operation(name, file_path)
        
        # Log memory usage before tool execution
        if psutil is not None:
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.debug(
                json.dumps(
                    {
                        "event": "memory_usage",
                        "rss": mem_info.rss,
                        "vms": mem_info.vms,
                    },
                ),
            )

        logger.debug(json.dumps({"event": "tool_execution_start", "tool": name}))

        # Route to the appropriate handler
        if name == "batch_convert":
            result = await handle_batch_convert(arguments)
        elif name == "batch_pages_convert":
            result = await handle_batch_pages_convert(arguments)
        elif name == "single_convert":
            if "fpath" in arguments and "file_path" not in arguments:
                arguments["file_path"] = arguments.pop("fpath")
            result = await handle_single_convert(arguments)
        elif name == "chunk_convert":
            result = await handle_chunk_convert(arguments)
        elif name == "start_server":
            result = await handle_start_server(arguments)
        else:
            error_msg = f"Unknown tool: {name!s}"
            logger.error(
                json.dumps(
                    {
                        "event": "error",
                        "error_code": "UNKNOWN_TOOL",
                        "message": error_msg,
                    },
                ),
            )
            if metrics_collector and job_id:
                metrics_collector.end_operation(job_id, success=False, error_message=error_msg)
            return {
                "success": False,
                "error_code": "UNKNOWN_TOOL",
                "error": error_msg,
                "message": f"Unknown tool: {name!s}",
            }

        # End operation tracking with success
        if metrics_collector and job_id:
            pages_processed = None
            if isinstance(result, dict) and "pages_processed" in result:
                pages_processed = result["pages_processed"]
            metrics_collector.end_operation(job_id, success=True, pages_processed=pages_processed)

        logger.info(json.dumps({"event": "tool_success", "tool": name}))
        logger.debug(
            json.dumps({"event": "tool_result", "result": result}, default=str),
        )

        if isinstance(result, (dict, str)):
            return result
        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        error_msg = f"Error in tool {name!s}: {e!s}"
        
        # End operation tracking with failure
        if metrics_collector and job_id:
            metrics_collector.end_operation(job_id, success=False, error_message=error_msg)
        
        logger.exception(
            json.dumps(
                {
                    "event": "error",
                    "error_code": "TOOL_EXECUTION_ERROR",
                    "tool": name,
                    "message": error_msg,
                    "traceback": traceback.format_exc(),
                },
            ),
        )
        return {
            "success": False,
            "error_code": "TOOL_EXECUTION_ERROR",
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "message": f"Failed to execute {name!s}: {e!s}",
        }


def get_log_level() -> str:
    """Get the log level from the environment variable LOG_LEVEL or default to INFO.

    Returns:
        The log level as a string.

    """
    return os.environ.get("LOG_LEVEL", "INFO").upper()


def log_startup_info(logger: "Logger") -> None:
    """Log startup information for the Marker MCP server.

    Args:
        logger: The logger instance to use for logging.

    """
    logger.info("=" * 40)
    logger.info("Starting Marker MCP server...")
    logger.info("Python: %s", sys.version)
    logger.info("Executable: %s", sys.executable)
    logger.info("Working directory: %s", Path.cwd())
    logger.info("Log level: %s", get_log_level())
    try:
        if torch is not None:
            logger.info("PyTorch version: %s", torch.__version__)
            logger.info("CUDA available: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                logger.info(
                    "CUDA device count: %s", torch.cuda.device_count(),
                )
                for i in range(torch.cuda.device_count()):
                    logger.info(
                        "  Device %s: %s", i, torch.cuda.get_device_name(i),
                    )
    except ImportError:
        logger.warning("PyTorch not available")


def configure_mcp_tools(mcp: FastMCP) -> None:
    """Configure MCP tools by registering them with the FastMCP instance.

    Args:
        mcp: The FastMCP instance to register tools with.

    """
    @mcp.tool()
    async def batch_convert(**kwargs: object) -> dict[str, Any]:
        """Batch convert all files in a folder using Marker with full argumentsupport.

        Args:
            **kwargs: Arguments for batch conversion.

        Returns:
            dict[str, Any]: Result of the batch conversion.

        """
        # Handle nested kwargs structure (common MCP pattern)
        actual_kwargs = kwargs
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            actual_kwargs = kwargs['kwargs']
        
        arguments = {
            "in_folder": actual_kwargs.get("folder_path"),
            "output_dir": actual_kwargs.get("output_dir") or None,
            "chunk_idx": actual_kwargs.get("chunk_idx", 0),
            "num_chunks": actual_kwargs.get("num_chunks", 1),
            "max_files": actual_kwargs.get("max_files")
            if safe_int(actual_kwargs.get("max_files", 0)) > 0 else None,
            "workers": actual_kwargs.get("workers", 5),
            "skip_existing": actual_kwargs.get("skip_existing", False),
            "debug_print": actual_kwargs.get("debug_print", False),
            "max_tasks_per_worker": actual_kwargs.get("max_tasks_per_worker", 10),
            "debug": actual_kwargs.get("debug", False),
            "output_format": actual_kwargs.get("output_format", "markdown"),
            "processors": actual_kwargs.get("processors") or None,
            "config_json": actual_kwargs.get("config_json") or None,
            "disable_multiprocessing": actual_kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": actual_kwargs.get("disable_image_extraction", False),
            "page_range": actual_kwargs.get("page_range") or None,
            "converter_cls": actual_kwargs.get("converter_cls") or None,
            "llm_service": actual_kwargs.get("llm_service") or None,
            "use_llm": actual_kwargs.get("use_llm", False),
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await handle_tool_call("batch_convert", arguments)
        return (
            result
            if isinstance(result, dict)
            else {"success": False, "error": str(result)}
        )

    @mcp.tool()
    async def single_convert(**kwargs: object) -> dict[str, Any]:
        """Convert a single PDF file using Marker with full CLI argument support.

        Args:
            **kwargs: Arguments for single file conversion.

        Returns:
            dict[str, Any]: Result of the single file conversion.

        """
        # Handle nested kwargs structure (common MCP pattern)
        actual_kwargs = kwargs
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            actual_kwargs = kwargs['kwargs']
        
        arguments = {
            "file_path": actual_kwargs.get("file_path"),
            "output_dir": actual_kwargs.get("output_dir") or None,
            "output_path": actual_kwargs.get("output_path") or None,
            "device": actual_kwargs.get("device") or None,
            "debug": actual_kwargs.get("debug", False),
            "output_format": actual_kwargs.get("output_format", "markdown"),
            "processors": actual_kwargs.get("processors") or None,
            "config_json": actual_kwargs.get("config_json") or None,
            "disable_multiprocessing": actual_kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": actual_kwargs.get("disable_image_extraction", False),
            "page_range": actual_kwargs.get("page_range") or None,
            "converter_cls": actual_kwargs.get("converter_cls") or None,
            "llm_service": actual_kwargs.get("llm_service") or None,
            "use_llm": actual_kwargs.get("use_llm", False),
            "max_pages": actual_kwargs.get("max_pages")
            if safe_int(actual_kwargs.get("max_pages", 0)) > 0 else None,
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await handle_tool_call("single_convert", arguments)
        return (
            result
            if isinstance(result, dict)
            else {"success": False, "error": str(result)}
        )

    @mcp.tool()
    async def chunk_convert(**kwargs: object) -> dict[str, Any]:
        """Convert PDFs in a folder using chunked processing.

        Args:
            **kwargs: Arguments for chunked conversion.

        Returns:
            dict[str, Any]: Result of the chunked conversion.

        """
        # Handle nested kwargs structure (common MCP pattern)
        actual_kwargs = kwargs
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            actual_kwargs = kwargs['kwargs']
        
        arguments = {
            "in_folder": actual_kwargs.get("in_folder"),
            "out_folder": actual_kwargs.get("out_folder") or None,
            "chunk_size": actual_kwargs.get("chunk_size", 10),
            "debug": actual_kwargs.get("debug", False),
            "output_format": actual_kwargs.get("output_format", "markdown"),
            "processors": actual_kwargs.get("processors") or None,
            "config_json": actual_kwargs.get("config_json") or None,
            "disable_multiprocessing": actual_kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": actual_kwargs.get("disable_image_extraction", False),
            "page_range": actual_kwargs.get("page_range") or None,
            "converter_cls": actual_kwargs.get("converter_cls") or None,
            "llm_service": actual_kwargs.get("llm_service") or None,
            "use_llm": actual_kwargs.get("use_llm", False),
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await handle_tool_call("chunk_convert", arguments)
        return (
            result
            if isinstance(result, dict)
            else {"success": False, "error": str(result)}
        )

    @mcp.tool()
    async def batch_pages_convert(**kwargs: object) -> dict[str, Any]:
        """Convert a single PDF by processing pages in chunks and stitching.

        results together.

        Args:
            **kwargs: Arguments for batch pages conversion.

        Returns:
            dict[str, Any]: Result of the batch pages conversion.

        """
        debug_log_file = "/Users/moatasimfarooque/Downloads/marker-1.7.3/debug_mcp_server_batch_pages_convert_kwargs.txt"
        
        def _log_debug(message):
            logger.info(message)
            try:
                with open(debug_log_file, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            except Exception as e:
                logger.error(f"Failed to write to debug log file {debug_log_file}: {e}")

        _log_debug(f"batch_pages_convert raw kwargs received: {kwargs}")
        _log_debug(f"kwargs type: {type(kwargs)}")
        _log_debug(f"kwargs keys: {list(kwargs.keys()) if hasattr(kwargs, 'keys') else 'no keys method'}")

        # Create debug file immediately (this part was for a different file, adapting for the new one)
        try:
            import json
            debug_info = {
                "function_called": "batch_pages_convert",
                "kwargs_type": str(type(kwargs)),
                "kwargs_keys": list(kwargs.keys()) if hasattr(kwargs, 'keys') else 'no keys method',
                "kwargs_content_str": str(kwargs), # Avoid direct serialization if it causes issues
                "kwargs_repr": repr(kwargs)
            }
            # Overwrite the main debug_mcp_call.txt with initial raw data for this specific call
            with open("debug_mcp_call.txt", "w") as f: # This is the general MCP call debug
                f.write(json.dumps(debug_info, indent=2))
        except Exception as e:
            _log_debug(f"Error writing to debug_mcp_call.txt: {e}")
        
        actual_kwargs = kwargs
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            _log_debug("Found nested 'kwargs' structure. Extracting parameters from kwargs['kwargs'].")
            actual_kwargs = kwargs['kwargs']
            _log_debug(f"Extracted actual_kwargs: {actual_kwargs}")
            _log_debug(f"actual_kwargs type: {type(actual_kwargs)}")
            _log_debug(f"actual_kwargs keys: {list(actual_kwargs.keys()) if hasattr(actual_kwargs, 'keys') else 'no keys method'}")
        else:
            _log_debug("No nested 'kwargs' structure found. Using raw kwargs as actual_kwargs.")

        file_path = None
        # Attempt 1: Get 'file_path'
        file_path = actual_kwargs.get("file_path")
        _log_debug(f"Attempt 1: actual_kwargs.get('file_path'): {file_path}")

        # Attempt 2: Get 'fpath'
        if file_path is None:
            file_path = actual_kwargs.get("fpath")
            _log_debug(f"Attempt 2: actual_kwargs.get('fpath'): {file_path}")

        # Attempt 3: Get 'pdf_path'
        if file_path is None:
            file_path = actual_kwargs.get("pdf_path")
            _log_debug(f"Attempt 3: actual_kwargs.get('pdf_path'): {file_path}")

        # Attempt 4: Get 'input_file'
        if file_path is None:
            file_path = actual_kwargs.get("input_file")
            _log_debug(f"Attempt 4: actual_kwargs.get('input_file'): {file_path}")
        
        # Attempt 5: Iterate through keys for a .pdf file if still None
        if file_path is None:
            _log_debug("file_path is still None. Iterating through actual_kwargs items...")
            for key, value in actual_kwargs.items():
                _log_debug(f"  Checking key: '{key}', value: '{value}', type: {type(value)}")
                if isinstance(value, str) and value.endswith('.pdf'):
                    _log_debug(f"Found potential PDF path in actual_kwargs['{key}']: {value}")
                    file_path = value
                    break
            if file_path is None:
                _log_debug("No .pdf file found by iterating through actual_kwargs.")
        
        _log_debug(f"Final file_path value before passing to arguments dict: {file_path}")
        
        arguments = {
            "file_path": file_path,
            "output_dir": actual_kwargs.get("output_dir") or None,
            "pages_per_chunk": actual_kwargs.get("pages_per_chunk", 5),
            "combine_output": actual_kwargs.get("combine_output", True),
            "debug": actual_kwargs.get("debug", False),
            "output_format": actual_kwargs.get("output_format", "markdown"),
            "processors": actual_kwargs.get("processors") or None,
            "config_json": actual_kwargs.get("config_json") or None,
            "disable_multiprocessing": actual_kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": actual_kwargs.get("disable_image_extraction", False),
            "converter_cls": actual_kwargs.get("converter_cls") or None,
            "llm_service": actual_kwargs.get("llm_service") or None,
            "use_llm": actual_kwargs.get("use_llm", False),
        }
        
        # Debug logging to see arguments before filtering
        logger.info(f"Arguments before filtering: {arguments}")
        
        # Don't filter out file_path even if it's None - let the handler deal with it
        arguments = {k: v for k, v in arguments.items() if k == "file_path" or v is not None}
        
        # Debug logging to see arguments after filtering
        logger.info(f"Arguments after filtering: {arguments}")
        
        result = await handle_tool_call("batch_pages_convert", arguments)
        return (
            result
            if isinstance(result, dict)
            else {"success": False, "error": str(result)}
        )

    @mcp.tool()
    async def start_server(**kwargs: object) -> dict[str, Any]:
        """Start the Marker FastAPI server for API-based PDF conversion.

        Args:
            **kwargs: Arguments for starting the server.

        Returns:
            dict[str, Any]: Result of the server start attempt.

        """
        # Handle nested kwargs structure (common MCP pattern)
        actual_kwargs = kwargs
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
            actual_kwargs = kwargs['kwargs']
        
        arguments = {
            "host": actual_kwargs.get("host", "127.0.0.1"),
            "port": actual_kwargs.get("port", 8000),
        }
        result = await handle_tool_call("start_server", arguments)
        return (
            result
            if isinstance(result, dict)
            else {"success": False, "error": str(result)}
        )

    @mcp.tool()
    async def get_system_health(**kwargs: object) -> dict[str, Any]:
        """Get current system health and performance metrics.

        Returns:
            dict[str, Any]: System health information including resource usage and alerts.

        """
        try:
            metrics_collector = get_metrics_collector()
            if not metrics_collector:
                return {"error": "Monitoring not initialized"}
            
            health = metrics_collector.get_system_health()
            return {
                "success": True,
                "health": {
                    "status": health.status,
                    "memory_status": health.memory_status,
                    "processing_status": health.processing_status,
                    "active_jobs": health.active_jobs,
                    "queue_size": health.queue_size,
                    "alerts": health.alerts,
                    "timestamp": health.timestamp
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def get_metrics_summary(**kwargs: object) -> dict[str, Any]:
        """Get performance metrics summary for a specified time period.

        Args:
            **kwargs: Optional arguments including 'hours' for time period.

        Returns:
            dict[str, Any]: Metrics summary including resource usage and performance data.

        """
        try:
            # Handle nested kwargs structure (common MCP pattern)
            actual_kwargs = kwargs
            if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], dict):
                actual_kwargs = kwargs['kwargs']
            
            metrics_collector = get_metrics_collector()
            if not metrics_collector:
                return {"error": "Monitoring not initialized"}

            hours = safe_int(actual_kwargs.get("hours", 1))
            summary = metrics_collector.get_metrics_summary(hours)
            return {"success": True, "metrics": summary}
        except Exception as e:
            return {"success": False, "error": str(e)}


def print_help() -> None:
    """Print help information about the Marker MCP server."""
    help_text = (
        """
Marker MCP Server

This server provides MCP (Model Context Protocol) tools for PDF conversion using Marker.

USAGE:
    python -m marker_mcp_server.server [OPTIONS]

OPTIONS:
    --help, -h          Show this help message and exit
    --version           Show version information
    --debug             Enable debug logging

AVAILABLE MCP TOOLS:
    batch_convert       Convert multiple PDFs in a folder
    batch_pages_convert Convert a single PDF by processing pages in chunks
    single_convert      Convert a single PDF file
    chunk_convert       Convert PDFs in chunks for large folders
    start_server        Start the Marker FastAPI server

TOOL ARGUMENTS:
    Each tool supports various arguments. Key options include:

    Common Options:
        debug              Enable debug mode (saves debug images/data)
        output_format      Format: markdown, json, html (default: markdown)
        processors         Comma-separated list of processors
        config_json        Path to JSON config file
        page_range         Page range (e.g., "0,5-10,20")
        llm_service        LLM service to use
        use_llm            Enable LLM processing for higher quality

    batch_convert specific:
        folder_path        Input folder with PDFs (required)
        output_dir         Output directory
        chunk_idx          Chunk index for parallel processing
        num_chunks         Total number of chunks
        max_files          Maximum files to process
        workers            Number of worker processes
        skip_existing      Skip already converted files

    single_convert specific:
        pdf_path           Path to PDF file (required)
        output_path        Output file path
        device             Device to use (auto, cpu, cuda, mps)
        max_pages          Maximum pages to convert

EXAMPLES:
    # Start the MCP server
    python -m marker_mcp_server.server

    # Show help
    python -m marker_mcp_server.server --help

For more information about MCP and how to use this server with MCP clients,
visit: https://github.com/modelcontextprotocol/
"""
    )
    logger.info(help_text)


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments.

    Returns:
        tuple: Parsed arguments and unknown arguments.
    """
    parser = argparse.ArgumentParser(
        description="Marker MCP Server - PDF conversion tools via MCP",
        add_help=False,  # We'll handle help ourselves
    )

    parser.add_argument(
        "--help", "-h",
        action="store_true",
        help="Show help message and exit",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The host to bind the server to (default: 127.0.0.1)",
    )

    return parser.parse_known_args()


async def run_dashboard(host: str, port: int):
    """Run the dashboard server."""
    logger.info(f"Starting dashboard server on {host}:{port}")
    try:
        config = uvicorn.Config(
            app=dashboard_app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        logger.error(f"Dashboard server error: {e}")
        raise

async def _start_dashboard_safely(server: uvicorn.Server, port: int) -> None:
    """Safely start the dashboard server without crashing the main process.
    
    Args:
        server: The uvicorn server instance
        port: The port to start on
    """
    # Monkey patch sys.exit to prevent uvicorn from crashing our process
    original_exit = sys.exit
    
    def safe_exit(code=0):
        """Replacement for sys.exit that logs instead of exiting."""
        logger.warning(f"Dashboard server attempted to exit with code {code} (prevented)")
        raise SystemExit(code)
    
    try:
        # Temporarily replace sys.exit
        sys.exit = safe_exit
        logger.info(f"Starting dashboard server on port {port}...")
        await server.serve()
        logger.info(f"Dashboard server started successfully on port {port}")
    except SystemExit as e:
        # uvicorn calls sys.exit() on binding errors, catch it here
        logger.warning(f"Dashboard server exited due to binding error on port {port} (exit code: {e.code})")
        raise  # Re-raise to let the caller know it failed
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Dashboard port {port} is already in use")
        else:
            logger.warning(f"Dashboard server OS error: {e}")
        raise  # Re-raise to let the caller know it failed
    except Exception as e:
        logger.warning(f"Dashboard server error: {e}")
        raise  # Re-raise to let the caller know it failed
    finally:
        # Always restore the original sys.exit
        sys.exit = original_exit


# Global variables to track dashboard server state
_dashboard_server = None
_dashboard_task = None

async def main_async():
    """Run the Marker MCP server main loop with async support."""
    global _dashboard_server, _dashboard_task
    
    args, unknown = parse_arguments()

    # Set log level from environment variable or command line
    log_level = get_log_level()
    if args.verbose or args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    logger.setLevel(log_level)

    logger.info("Starting Marker MCP Server...")
    log_startup_info(logger)

    # Initialize configuration
    config = Config()
    
    # Ensure required directories exist
    metrics_dir = os.path.expanduser("~/.cache/marker-mcp/metrics")
    logs_dir = os.path.expanduser("~/.cache/marker-mcp/logs")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Initialize monitoring with the config
    metrics_collector = None
    try:
        # Pass the config's _config attribute which contains the AppConfig object
        initialize_monitoring(config._config)
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            # Start metrics collection in the background
            asyncio.create_task(metrics_collector._collect_metrics_loop())
            logger.info("Monitoring system initialized")
        else:
            logger.warning("Failed to initialize metrics collector")
    except Exception as e:
        logger.error(f"Failed to initialize monitoring: {e}", exc_info=True)
    
    # Get server host and port from config
    server_host = '127.0.0.1'
    server_port = 8000
    
    # Try to get server config from the Config object
    try:
        if hasattr(config, '_config') and hasattr(config._config, 'server'):
            server_config = config._config.server
            if hasattr(server_config, 'host'):
                server_host = server_config.host
            if hasattr(server_config, 'port'):
                server_port = server_config.port
    except Exception as e:
        logger.debug(f"Could not access server config: {e}, using defaults")
    
    # Start dashboard in the background if not already running
    dashboard_port = server_port + 1
    
    # Check if dashboard should be started
    dashboard_started = False
    dashboard_host = "127.0.0.1"  # Use localhost for better compatibility
    
    if _dashboard_server is None:
        # First, check if the port is available
        if not is_port_available(dashboard_host, dashboard_port):
            logger.warning(f"Dashboard port {dashboard_port} is already in use, trying to find an available port...")
            available_port = find_available_port(dashboard_port, dashboard_host, 10)
            if available_port != -1:
                dashboard_port = available_port
                logger.info(f"Using alternative port {dashboard_port} for dashboard")
            else:
                logger.warning("No available ports found for dashboard, skipping dashboard startup")
                dashboard_port = None
        
        if dashboard_port is not None:
            logger.info(f"Starting dashboard server on {dashboard_host}:{dashboard_port}...")
            try:
                # Create uvicorn config with proper error handling
                dashboard_config = uvicorn.Config(
                    app=dashboard_app,
                    host=dashboard_host,
                    port=dashboard_port,
                    log_level="error",  # Reduce log noise
                    reload=False,
                    log_config=None,
                    access_log=False
                )
                _dashboard_server = uvicorn.Server(dashboard_config)
                
                # Start the server in a task that won't crash the main process
                _dashboard_task = asyncio.create_task(_start_dashboard_safely(_dashboard_server, dashboard_port))
                
                # Give it more time to start and check multiple times
                for attempt in range(5):
                    await asyncio.sleep(0.5)
                    # Check if it's actually running (if port is no longer available, server started)
                    if not is_port_available(dashboard_host, dashboard_port):
                        logger.info(f"Dashboard server started successfully on port {dashboard_port}")
                        logger.info(f"Dashboard available at http://{dashboard_host}:{dashboard_port}")
                        dashboard_started = True
                        break
                    
                    # Check if the task failed
                    if _dashboard_task.done():
                        try:
                            await _dashboard_task  # This will raise the exception if there was one
                        except Exception as task_error:
                            logger.warning(f"Dashboard task failed: {task_error}")
                            break
                
                if not dashboard_started:
                    logger.warning(f"Dashboard did not start successfully on port {dashboard_port}")
                
            except Exception as e:
                logger.warning(f"Failed to start dashboard server: {e}")
                if _dashboard_task:
                    _dashboard_task.cancel()
                    try:
                        await _dashboard_task
                    except (asyncio.CancelledError, Exception):
                        pass
                _dashboard_server = None
                _dashboard_task = None
    else:
        logger.info("Dashboard is already running")
        dashboard_started = True
    
    if not dashboard_started:
        logger.info("Continuing without dashboard - MCP server functionality will not be affected")
    
    # Start the MCP server
    try:
        # Create MCP server
        mcp = FastMCP(
            name="Marker MCP Server",
            version="1.0.0",
            description="MCP server for Marker conversion tools",
            config=config,  # Pass the config object directly
        )
        
        # Configure MCP tools
        configure_mcp_tools(mcp)
        
        # Get host from config or use default
        server_host = '127.0.0.1'
        try:
            if hasattr(config, '_config') and hasattr(config._config, 'server') and hasattr(config._config.server, 'host'):
                server_host = config._config.server.host
        except Exception:
            pass  # Use default
        
        # Start the MCP server
        logger.info(f"Starting MCP server on {server_host}:{server_port}")
        await mcp.run_stdio_async()
        
    except asyncio.CancelledError:
        logger.info("MCP server task cancelled")
    except Exception as e:
        logger.error(f"Error in MCP server: {e}", exc_info=True)
        raise
    finally:
        # Clean up dashboard server
        if _dashboard_server is not None:
            logger.info("Shutting down dashboard server...")
            _dashboard_server.should_exit = True
            if _dashboard_task and not _dashboard_task.done():
                _dashboard_task.cancel()
                try:
                    await _dashboard_task
                except (asyncio.CancelledError, Exception) as e:
                    logger.debug(f"Dashboard shutdown: {e}")
            _dashboard_server = None
            _dashboard_task = None
        
        # Clean up monitoring
        if metrics_collector is not None:
            logger.info("Shutting down monitoring...")
            await shutdown_monitoring()
        logger.info("Server stopped")


async def main() -> None:
    """Run the Marker MCP server main loop."""
    try:
        # Simply call main_async which handles everything
        await main_async()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        args, unknown = parse_arguments()
        if args.help:
            print_help()
            sys.exit(0)
        if args.version:
            logger.info("Marker MCP Server")
            try:
                import marker
                version = getattr(marker, "__version__", "unknown")
                logger.info("Marker version: %s", version)
            except ImportError:
                logger.info("Marker version: unknown")
            sys.exit(0)
        if args.debug:
            os.environ["LOG_LEVEL"] = "DEBUG"
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
