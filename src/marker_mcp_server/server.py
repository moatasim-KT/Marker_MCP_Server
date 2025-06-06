# Copyright (c) 2024 Marker MCP Server Authors
"""Marker MCP Server.

This module implements an MCP server for interacting with Marker conversion tools.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import FastMCP

from .tools import (
    handle_batch_convert,
    handle_batch_pages_convert,
    handle_chunk_convert,
    handle_single_convert,
    handle_start_server,
)
from .monitoring import initialize_monitoring, get_metrics_collector, shutdown_monitoring
from .security import create_security_validator
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
        arguments = {
            "in_folder": kwargs.get("folder_path"),
            "output_dir": kwargs.get("output_dir") or None,
            "chunk_idx": kwargs.get("chunk_idx", 0),
            "num_chunks": kwargs.get("num_chunks", 1),
            "max_files": kwargs.get("max_files")
            if safe_int(kwargs.get("max_files", 0)) > 0 else None,
            "workers": kwargs.get("workers", 5),
            "skip_existing": kwargs.get("skip_existing", False),
            "debug_print": kwargs.get("debug_print", False),
            "max_tasks_per_worker": kwargs.get("max_tasks_per_worker", 10),
            "debug": kwargs.get("debug", False),
            "output_format": kwargs.get("output_format", "markdown"),
            "processors": kwargs.get("processors") or None,
            "config_json": kwargs.get("config_json") or None,
            "disable_multiprocessing": kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": kwargs.get("disable_image_extraction", False),
            "page_range": kwargs.get("page_range") or None,
            "converter_cls": kwargs.get("converter_cls") or None,
            "llm_service": kwargs.get("llm_service") or None,
            "use_llm": kwargs.get("use_llm", False),
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
        arguments = {
            "file_path": kwargs.get("file_path"),
            "output_dir": kwargs.get("output_dir") or None,
            "output_path": kwargs.get("output_path") or None,
            "device": kwargs.get("device") or None,
            "debug": kwargs.get("debug", False),
            "output_format": kwargs.get("output_format", "markdown"),
            "processors": kwargs.get("processors") or None,
            "config_json": kwargs.get("config_json") or None,
            "disable_multiprocessing": kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": kwargs.get("disable_image_extraction", False),
            "page_range": kwargs.get("page_range") or None,
            "converter_cls": kwargs.get("converter_cls") or None,
            "llm_service": kwargs.get("llm_service") or None,
            "use_llm": kwargs.get("use_llm", False),
            "max_pages": kwargs.get("max_pages")
            if safe_int(kwargs.get("max_pages", 0)) > 0 else None,
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
        arguments = {
            "in_folder": kwargs.get("in_folder"),
            "out_folder": kwargs.get("out_folder") or None,
            "chunk_size": kwargs.get("chunk_size", 10),
            "debug": kwargs.get("debug", False),
            "output_format": kwargs.get("output_format", "markdown"),
            "processors": kwargs.get("processors") or None,
            "config_json": kwargs.get("config_json") or None,
            "disable_multiprocessing": kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": kwargs.get("disable_image_extraction", False),
            "page_range": kwargs.get("page_range") or None,
            "converter_cls": kwargs.get("converter_cls") or None,
            "llm_service": kwargs.get("llm_service") or None,
            "use_llm": kwargs.get("use_llm", False),
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
        arguments = {
            "file_path": kwargs.get("file_path"),
            "output_dir": kwargs.get("output_dir") or None,
            "pages_per_chunk": kwargs.get("pages_per_chunk", 5),
            "combine_output": kwargs.get("combine_output", True),
            "debug": kwargs.get("debug", False),
            "output_format": kwargs.get("output_format", "markdown"),
            "processors": kwargs.get("processors") or None,
            "config_json": kwargs.get("config_json") or None,
            "disable_multiprocessing": kwargs.get("disable_multiprocessing", False),
            "disable_image_extraction": kwargs.get("disable_image_extraction", False),
            "converter_cls": kwargs.get("converter_cls") or None,
            "llm_service": kwargs.get("llm_service") or None,
            "use_llm": kwargs.get("use_llm", False),
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
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
        arguments = {
            "host": kwargs.get("host", "127.0.0.1"),
            "port": kwargs.get("port", 8000),
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
            metrics_collector = get_metrics_collector()
            if not metrics_collector:
                return {"error": "Monitoring not initialized"}

            hours = int(kwargs.get("hours", 1))
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

    return parser.parse_known_args()


async def main() -> None:
    """Run the Marker MCP server main loop."""
    try:
        # Configure logging
        log_level = get_log_level()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
        )
        log_startup_info(logger)
        
        # Initialize configuration
        config = Config()
        
        # Initialize monitoring
        metrics_collector = initialize_monitoring(config._config)
        await metrics_collector.start()
        logger.info("Monitoring system initialized")
        
        # Initialize security validator
        create_security_validator(config._config)
        logger.info("Security validator initialized")
        
        mcp = FastMCP("Marker MCP Server")
        configure_mcp_tools(mcp)
        logger.info("FastMCP server configured successfully")
        logger.info(
            "Available tools: batch_convert, batch_pages_convert, single_convert, "
            "chunk_convert, start_server, get_system_health, get_metrics_summary",
        )
        logger.info("-" * 40)
        logger.info("Waiting for connections...")
        logger.debug("Starting FastMCP server with stdio transport...")
        await mcp.run_stdio_async()
    except asyncio.CancelledError:
        logger.info("Server shutdown requested")
    except Exception as e:
        error_msg = f"Fatal error in main: {e!s}\n{traceback.format_exc()}"
        logger.exception(error_msg)
        sys.stderr.write(f"{error_msg}\n")
        sys.exit(1)
    finally:
        # Cleanup
        await shutdown_monitoring()
        logger.info("Marker MCP server shutting down")
        logger.info("=" * 40)


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
        logger.info("Server interrupted by user")
        sys.exit(0)
    except Exception:
        logger.exception(
            "Unhandled exception occurred:\n%s", traceback.format_exc(),
        )
        sys.exit(1)
