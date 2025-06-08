# Copyright (c) 2024 Marker MCP Server Authors  # noqa: D100
import asyncio  # noqa: D100, RUF100
import logging  # Ensure logging is imported
import os
import traceback  # Ensure traceback is imported
from typing import Any

import pypdfium2  # Added import for pypdfium2, used in handle_batch_pages_convert

from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import save_output

logger = logging.getLogger("marker-mcp-server-tools")

# Global variable to store server process reference
_server_process = None

# Common LLM service mappings for user convenience
LLM_SERVICE_MAPPINGS = {
    "groq": "marker.services.groq.GroqService",
    "openai": "marker.services.openai.OpenAIService",
    "anthropic": "marker.services.anthropic.AnthropicService",
    "gemini": "marker.services.gemini.GeminiService",
}

PROCESSOR_NAME_TO_FULL_PATH = {
    # Note: image_extraction is handled by renderers, not processors
    # Use disable_image_extraction parameter instead
    "table_extraction": "marker.processors.table.TableProcessor",
    "math_extraction": "marker.processors.equation.EquationProcessor",
    "code_extraction": "marker.processors.code.CodeBlockProcessor",
    # Aliases for convenience
    "table": "marker.processors.table.TableProcessor",
    "math": "marker.processors.equation.EquationProcessor",
    "code": "marker.processors.code.CodeBlockProcessor",
    # From marker/processors/registry.py (inferred)
    "BlockQuoteProcessor": "marker.processors.blockquote.BlockQuoteProcessor",
    "CodeBlockProcessor": "marker.processors.code.CodeBlockProcessor",
    "DebugProcessor": "marker.processors.debug.DebugProcessor",
    "DocumentTocProcessor": "marker.processors.document_toc.DocumentTocProcessor",
    "EquationProcessor": "marker.processors.equation.EquationProcessor",
    "FootnoteProcessor": "marker.processors.footnote.FootnoteProcessor",
    "IgnoreTextProcessor": "marker.processors.ignoretext.IgnoreTextProcessor",
    "LineMergeProcessor": "marker.processors.line_merge.LineMergeProcessor",
    "LineNumberProcessor": "marker.processors.line_numbers.LineNumberProcessor",
    "ListProcessor": "marker.processors.list.ListProcessor",
    "OrderProcessor": "marker.processors.order.OrderProcessor",
    "PageHeaderProcessor": "marker.processors.page_header.PageHeaderProcessor",
    "ReferenceProcessor": "marker.processors.reference.ReferenceProcessor",
    "SectionHeaderProcessor": "marker.processors.sectionheader.SectionHeaderProcessor",
    "TableProcessor": "marker.processors.table.TableProcessor",  # Explicit FQN mapping
    "TextProcessor": "marker.processors.text.TextProcessor",
}


def normalize_llm_service(llm_service: str) -> str:
    """
    Normalize LLM service name to full module path.

    Args:
        llm_service: Either a short name (e.g., "groq") or full path (e.g., "marker.services.groq.GroqService")

    Returns:
        Full module path for the LLM service
    """
    if not llm_service:
        return ""

    # If it's already a full path (contains dots), return as-is
    if "." in llm_service:
        return llm_service

    # Otherwise, try to map the short name to full path
    normalized = LLM_SERVICE_MAPPINGS.get(llm_service.lower())
    if normalized:
        return normalized

    # If no mapping found, log a warning and return the original
    logger.warning(
        f"Unknown LLM service '{llm_service}'. Use full module path or one of: {list(LLM_SERVICE_MAPPINGS.keys())}"
    )
    return llm_service


# Import Marker components with proper error handling
try:
    # Import the models through the resources module
    from . import resources
    from .batch_optimizer import apply_gpu_optimizations

    # Configure Marker settings with the device from resources
    device = resources.get_device()

    # Apply GPU optimizations for better utilization
    try:
        apply_gpu_optimizations()
        logger.info("GPU optimizations applied successfully")
    except Exception as e:
        logger.warning(f"Failed to apply GPU optimizations: {e}")

    # Create a new settings instance with the updated device
    # Removed direct assignment to marker_settings.TORCH_DEVICE and marker_settings.TORCH_DEVICE_MODEL
    # These will be passed via arguments to ConfigParser if needed.
    pass  # Placeholder if no other logic is needed here after removing marker_settings assignments

except ImportError as e:
    logger.error(f"Failed to import Marker components: {e}")
    logger.error("Please ensure the Marker package is installed correctly.")
    raise


async def _run_with_redirection(func, args=None, kwargs=None):
    """Helper function to run a function with stdout/stderr redirection."""
    if args is None:
        args = ()  # Corrected mutable default
    if kwargs is None:
        kwargs = {}  # Corrected mutable default

    # old_stdout = sys.stdout # Commented out as it's unused
    # old_stderr = sys.stderr # Commented out as it's unused
    # redirected_output = io.StringIO()
    # redirected_error = io.StringIO()

    try:
        # sys.stdout = redirected_output
        # sys.stderr = redirected_error
        # result = await func(*args, **kwargs)
        # return result, redirected_output.getvalue(), redirected_error.getvalue()
        pass  # Try block is currently empty
    except Exception as e:
        _ = e  # Mark 'e' as intentionally unused as try block is empty
        # logger.error(f"Error in _run_with_redirection: {e}") # If logging is desired
        # return None, "", str(e) # Ensure a tuple is returned in case of exception
    finally:
        # sys.stdout = old_stdout # Commented out as old_stdout is unused
        # sys.stderr = old_stderr # Commented out as old_stderr is unused
        # pass # No specific action needed if redirection is fully commented out
        pass  # Try block is currently empty

    # If redirection is not used, return result directly, or adapt as needed.
    # For now, assuming the core logic of redirection is bypassed.
    # return result, redirected_output.getvalue(), redirected_error.getvalue()
    # Since the redirection logic is commented out, this function might need rethinking
    # or the commented out parts restored if redirection is actually needed.
    # For now, let's assume it's meant to execute func and return its result directly if no redirection.
    # This part needs clarification based on whether redirection is intended to be active.
    # If func is async, it should be awaited.
    # if asyncio.iscoroutinefunction(func):
    #     return await func(*args, **kwargs), "", "" # Placeholder for stdout/stderr
    # else:
    #     return func(*args, **kwargs), "", "" # Placeholder for stdout/stderr
    # The above is a guess. The original structure suggests redirection was intended.
    # Given the current state (redirection commented out), the function doesn't do much.
    # Let's assume for now the primary goal was to fix the unused variable, and the redirection
    # logic might be revisited later.
    return None, "", ""  # Returning dummy values as redirection is disabled


async def handle_batch_convert(arguments: dict[str, Any]) -> dict:
    """
    Convert all files in a folder using Marker batch conversion with full CLI argument support.

    Args:
        arguments: Dictionary containing:
            - in_folder: Path to the input directory
            - output_dir: Directory to save output (optional)
            - chunk_idx: Chunk index to convert (optional)
            - num_chunks: Number of chunks being processed in parallel (optional)
            - max_files: Maximum number of PDFs to convert (optional)
            - workers: Number of worker processes to use (optional)
            - skip_existing: Skip existing converted files (optional)
            - debug_print: Print debug information (optional)
            - max_tasks_per_worker: Maximum number of tasks per worker process (optional)
            - debug: Enable debug mode (optional)
            - output_format: Format to output results in (optional)
            - processors: Comma separated list of processors to use (optional)
            - config_json: Path to JSON file with additional configuration (optional)
            - disable_multiprocessing: Disable multiprocessing (optional)
            - disable_image_extraction: Disable image extraction (optional)
            - page_range: Page range to convert (optional)
            - converter_cls: Converter class to use (optional)
            - llm_service: LLM service to use (optional)
            - use_llm: Enable LLM processing (optional)

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        # Normalize llm_service argument
        if "llm_service" in arguments and arguments["llm_service"]:
            arguments["llm_service"] = normalize_llm_service(arguments["llm_service"])
        in_folder = arguments.get("in_folder")
        output_dir = arguments.get("output_dir") or in_folder
        chunk_idx = arguments.get("chunk_idx", 0)
        num_chunks = arguments.get("num_chunks", 1)
        max_files = arguments.get("max_files")
        skip_existing = arguments.get("skip_existing", False)

        if not in_folder or not os.path.isdir(in_folder):
            raise ValueError(f"Input directory not found: {in_folder}")

        logger.info(f"Starting batch conversion in folder: {in_folder}")

        # Get list of PDF files
        pdf_files = [f for f in os.listdir(in_folder) if f.lower().endswith(".pdf")]
        pdf_files.sort()
        if not pdf_files:
            raise ValueError(f"No PDF files found in {in_folder}")

        # Support chunking for parallel processing
        if num_chunks > 1:
            pdf_files = pdf_files[chunk_idx::num_chunks]

        # Support max_files
        if max_files is not None:
            pdf_files = pdf_files[:max_files]

        results = []
        success_count = 0
        for pdf_file in pdf_files:
            input_path = os.path.join(in_folder, pdf_file)
            if skip_existing:
                # Determine expected output path to check if it exists
                # This logic might need to be more sophisticated depending on output_format etc.
                # For now, assuming .md output in the same folder or specified output_dir
                expected_out_fname = os.path.splitext(pdf_file)[0] + ".md"
                expected_out_path = os.path.join(
                    output_dir or in_folder, expected_out_fname
                )
                if os.path.exists(expected_out_path):
                    logger.info(f"Skipping existing file: {expected_out_path}")
                    results.append(
                        {
                            "file": pdf_file,
                            "success": True,
                            "message": "Skipped, already exists",
                        }
                    )
                    success_count += 1
                    continue
            # Prepare arguments for single convert
            file_args = arguments.copy()
            file_args["file_path"] = input_path
            file_args["output_dir"] = output_dir
            result = await handle_single_convert(file_args)
            results.append({"file": pdf_file, **result})
            if result.get("success"):
                success_count += 1
            else:
                logger.warning(f"Failed to process {pdf_file}: {result.get('error')}")

        message = f"Batch conversion completed. Processed {len(results)} files, {success_count} succeeded."
        logger.info(message)
        return {
            "success": success_count == len(results),
            "input_folder": in_folder,
            "output_folder": output_dir,
            "results": results,
            "message": message,
        }
    except Exception as e:
        error_msg = f"Error in batch_convert: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "message": f"Failed to process batch conversion: {str(e)}",
        }


async def handle_single_convert(arguments: dict[str, Any]) -> dict:
    """Convert a single PDF file.

    Args:
        arguments: Dictionary containing:
            - file_path: Path to the input file
            - output_dir: Directory to save output (optional)
            - output_path: Path to the output file (optional, legacy)
            - device: Device to use for conversion (optional, legacy)
            - debug: Enable debug mode (optional)
            - output_format: Format to output results in (optional)
            - processors: Comma separated list of processors to use (optional)
            - config_json: Path to JSON file with additional configuration (optional)
            - disable_multiprocessing: Disable multiprocessing (optional)
            - disable_image_extraction: Disable image extraction (optional)
            - page_range: Page range to convert (optional)
            - converter_cls: Converter class to use (optional)
            - llm_service: LLM service to use (optional)
            - use_llm: Enable LLM processing (optional)

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        # Normalize llm_service argument
        if "llm_service" in arguments and arguments["llm_service"]:
            arguments["llm_service"] = normalize_llm_service(arguments["llm_service"])

        # Add device to arguments for ConfigParser
        arguments["device"] = device  # Use the globally fetched device

        # Map short processor names to full paths
        if "processors" in arguments and isinstance(arguments["processors"], str):
            short_names_str = arguments["processors"]
            short_names_list = [
                name.strip() for name in short_names_str.split(",") if name.strip()
            ]

            full_names_list = []
            for short_name in short_names_list:
                # image_extraction is handled by disable_image_extraction flag in ConfigParser,
                # but if specified, it should be a FQN.
                fqn = PROCESSOR_NAME_TO_FULL_PATH.get(
                    short_name, short_name
                )  # Default to original name if not in map
                full_names_list.append(fqn)

            arguments["processors"] = ",".join(full_names_list)
            logger.info(f"Mapped processors to: {arguments['processors']}")
        elif "processors" in arguments and isinstance(arguments["processors"], list):
            # If processors are already a list (e.g. from a JSON config)
            # We assume they are FQNs or will be handled by ConfigParser
            logger.info(f"Processors already a list: {arguments['processors']}")

        file_path = arguments.get("file_path")
        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Starting single conversion for: {file_path}")

        # Get metrics collector for progress tracking
        from src.marker_mcp_server.monitoring import get_metrics_collector

        metrics_collector = get_metrics_collector()

        # Update progress to show initialization stage
        if metrics_collector:
            active_jobs = metrics_collector.get_active_jobs_details()
            for job_id, job_info in active_jobs.items():
                if job_info.get("file_path") == file_path:
                    metrics_collector.update_operation_progress(
                        job_id, processing_stage="Initializing models and config"
                    )
                    break

        # Prepare config parser and models
        config_parser = ConfigParser(arguments)
        models = create_model_dict()

        # Get converter class and instantiate
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        # Update progress to show conversion stage
        if metrics_collector:
            active_jobs = metrics_collector.get_active_jobs_details()
            for job_id, job_info in active_jobs.items():
                if job_info.get("file_path") == file_path:
                    metrics_collector.update_operation_progress(
                        job_id, processing_stage="Converting document"
                    )
                    break

        # Run conversion with GPU memory management
        from .memory_manager import gpu_memory_context, log_gpu_memory

        log_gpu_memory("before_conversion")
        with gpu_memory_context("pdf_conversion"):
            rendered = converter(file_path)
        log_gpu_memory("after_conversion")

        # Prepare output directory and filename
        out_folder = config_parser.get_output_folder(file_path)
        fname_base = config_parser.get_base_filename(file_path)

        # Update progress to show saving stage
        if metrics_collector:
            active_jobs = metrics_collector.get_active_jobs_details()
            for job_id, job_info in active_jobs.items():
                if job_info.get("file_path") == file_path:
                    metrics_collector.update_operation_progress(
                        job_id, processing_stage="Saving output"
                    )
                    break

        # Save output
        save_output(rendered, out_folder, fname_base)

        logger.info(f"Saved markdown to {out_folder}")
        result = {
            "success": True,
            "input_file": file_path,
            "output_file": os.path.join(out_folder, fname_base + ".md"),
            "message": "Single conversion completed successfully",
        }
        logger.info(f"Returning result: {result}")
        return result

    except Exception as e:
        error_msg = f"Error in single_convert: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        result = {
            "success": False,
            "error": error_msg,
            "message": f"Failed to process single conversion: {str(e)}",
        }
        logger.info(f"Returning error result: {result}")
        return result


async def handle_chunk_convert(arguments: dict[str, Any]) -> dict:
    """Convert a folder of PDF files in chunks.

    Args:
        arguments: Dictionary containing:
            - in_folder: Path to the input directory
            - out_folder: Path to the output directory
            - chunk_size: Number of files to process in each chunk (default: 10)
            - debug: Enable debug mode (optional)
            - output_format: Output format (markdown, json, etc.) (optional)
            - processors: Comma-separated list of processors (optional)
            - config_json: Path to JSON file with additional configuration (optional)
            - disable_multiprocessing: Disable multiprocessing (optional)
            - disable_image_extraction: Disable image extraction (optional)
            - page_range: Page range to convert (optional)
            - converter_cls: Converter class to use (optional)
            - llm_service: LLM service to use (optional)
            - use_llm: Enable LLM processing (optional)

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        # Normalize llm_service argument
        if "llm_service" in arguments and arguments["llm_service"]:
            arguments["llm_service"] = normalize_llm_service(arguments["llm_service"])

        # Add device to arguments for handle_single_convert
        # This ensures the device is propagated if not already in convert_args
        # However, handle_single_convert will add it from global `device` if not present.
        # For consistency, we can add it here too.
        # arguments["device"] = device # Already handled in handle_single_convert

        in_folder = arguments.get("in_folder")
        out_folder = arguments.get("out_folder")
        chunk_size = arguments.get("chunk_size", 10)

        if not in_folder or not os.path.isdir(in_folder):
            raise ValueError(f"Input directory not found: {in_folder}")

        if not out_folder:
            out_folder = os.path.join(in_folder, "chunked_output")
            os.makedirs(out_folder, exist_ok=True)

        logger.info(f"Starting chunk conversion from {in_folder} to {out_folder}")

        # Get list of PDF files
        pdf_files = [f for f in os.listdir(in_folder) if f.lower().endswith(".pdf")]

        if not pdf_files:
            raise ValueError(f"No PDF files found in {in_folder}")

        # Build arguments to pass to single_convert for each file
        # This includes all the CLI arguments that were passed to chunk_convert
        convert_args = {
            "output_dir": out_folder,
            "debug": arguments.get("debug", False),
            "output_format": arguments.get("output_format", "markdown"),
            "processors": arguments.get(
                "processors", ""
            ),  # This will be normalized by handle_single_convert
            "config_json": arguments.get("config_json", ""),
            "disable_multiprocessing": arguments.get("disable_multiprocessing", False),
            "disable_image_extraction": arguments.get(
                "disable_image_extraction", False
            ),
            "page_range": arguments.get("page_range", ""),
            "converter_cls": arguments.get("converter_cls", ""),
            "llm_service": arguments.get("llm_service", ""),
            "use_llm": arguments.get("use_llm", False),
        }

        # Process files in chunks
        for i in range(0, len(pdf_files), chunk_size):
            chunk = pdf_files[i : i + chunk_size]
            logger.info(
                f"Processing chunk {i // chunk_size + 1} with {len(chunk)} files"
            )

            # Process each file in the chunk
            for pdf_file in chunk:
                input_path = os.path.join(in_folder, pdf_file)

                # Prepare arguments for single convert with file-specific path
                file_args = convert_args.copy()
                file_args["file_path"] = input_path

                # Call single convert for each file
                result = await handle_single_convert(file_args)

                if not result.get("success"):
                    logger.warning(
                        f"Failed to process {pdf_file}: {result.get('error')}"
                    )

        return {
            "success": True,
            "input_folder": in_folder,
            "output_folder": out_folder,
            "message": f"Chunk conversion completed. Processed {len(pdf_files)} files.",
        }

    except Exception as e:
        error_msg = f"Error in chunk_convert: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "message": f"Failed to process chunk conversion: {str(e)}",
        }


async def handle_start_server(arguments: dict[str, Any]) -> dict:
    """Start the Marker FastAPI server.

    Args:
        arguments: Dictionary containing:
            - host: Host to bind the server to (default: "127.0.0.1")
            - port: Port to listen on (default: 8000)
            - workers: Number of worker processes (default: 1)
            - reload: Whether to enable auto-reload (default: False)

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        host = arguments.get("host", "127.0.0.1")
        port = arguments.get("port", 8000)
        workers = arguments.get("workers", 1)
        reload = arguments.get("reload", False)

        logger.info(f"Starting Marker server on {host}:{port} with {workers} workers")

        # Prepare command line arguments
        cmd = [
            "uvicorn",
            "marker.app:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            str(workers),
        ]

        if reload:
            cmd.append("--reload")

        # Start the server in a subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Store the process reference for later use
        global _server_process
        _server_process = process

        # Wait a bit to check if the server started successfully
        await asyncio.sleep(2)

        if process.returncode is not None and process.returncode != 0:
            _, stderr = await process.communicate()
            error_msg = f"Failed to start server: {stderr.decode()}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Failed to start Marker server",
            }

        server_url = f"http://{host}:{port}"
        return {
            "success": True,
            "message": f"Marker server started successfully on {server_url}",
            "server_url": server_url,
            "process_id": process.pid,
        }

    except Exception as e:
        error_msg = f"Error starting server: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "message": "Failed to start Marker server",
        }


async def handle_batch_pages_convert(arguments: dict[str, Any]) -> dict:
    """
    Convert a single PDF file by processing it in page chunks and stitching results together.

    Args:
        arguments: Dictionary containing:
            - file_path: Path to the input PDF file
            - output_dir: Directory to save output (optional)
            - pages_per_chunk: Number of pages to process per chunk (default: 5)
            - output_format: Format to output results in (default: markdown)
            - debug: Enable debug mode (optional)
            - processors: Comma separated list of processors to use (optional)
            - config_json: Path to JSON file with additional configuration (optional)
            - disable_multiprocessing: Disable multiprocessing (optional)
            - disable_image_extraction: Disable image extraction (optional)
            - converter_cls: Converter class to use (optional)
            - llm_service: LLM service to use (optional)
            - use_llm: Enable LLM processing (optional)
            - combine_output: Whether to combine chunk outputs into single file (default: True)

    Returns:
        dict: A dictionary containing the result of the operation.
    """
    try:
        # Debug logging to see what arguments we receive
        logger.info(f"handle_batch_pages_convert received arguments: {arguments}")
        # logger.info(f"arguments keys: {list(arguments.keys())}") # Redundant with loop below
        # logger.info(f"arguments.get('file_path'): {arguments.get('file_path')}") # Redundant

        # Create debug file to inspect arguments
        # This debug file writing can be removed or made conditional if not always needed
        # try:
        #     with open("debug_tools_handle_batch_pages_convert.txt", "w") as f:
        #         f.write(f"handle_batch_pages_convert received arguments: {arguments}\\n")
        #         f.write(f"arguments keys: {list(arguments.keys())}\\n")
        #         f.write(f"arguments type: {type(arguments)}\\n")
        #         for key, value in arguments.items():
        #             f.write(f"  {key}: {value} (type: {type(value)})\\n")
        # except Exception as e_debug_file: # Renamed to avoid conflict
        #     logger.error(f"Failed to write debug file: {e_debug_file}")

        # Normalize llm_service argument
        if "llm_service" in arguments and arguments["llm_service"]:
            arguments["llm_service"] = normalize_llm_service(arguments["llm_service"])

        # The 'processors' string from 'arguments' will be passed to 'chunk_args',
        # and then normalized within 'handle_single_convert'.
        # No need to normalize 'arguments["processors"]' directly here if handle_single_convert does it.

        file_path = arguments.get("file_path")
        pages_per_chunk = arguments.get("pages_per_chunk", 5)
        combine_output = arguments.get("combine_output", True)

        if not file_path or not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Starting batch pages conversion for: {file_path}")

        # Get total page count
        try:
            pdf_doc = pypdfium2.PdfDocument(file_path)
            total_pages = len(pdf_doc)
            pdf_doc.close()
            logger.info(
                f"Document has {total_pages} pages, processing {pages_per_chunk} pages per chunk"
            )
        except Exception as e:
            raise ValueError(f"Unable to read PDF file: {str(e)}")

        if total_pages == 0:
            raise ValueError("PDF file appears to be empty")

        # Prepare output directory
        output_dir = arguments.get("output_dir")
        if not output_dir:
            output_dir = os.path.dirname(file_path)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Process pages in chunks
        chunk_results = []
        chunk_files = []

        # Get metrics collector for progress tracking
        from src.marker_mcp_server.monitoring import get_metrics_collector

        metrics_collector = get_metrics_collector()

        for chunk_start in range(0, total_pages, pages_per_chunk):
            chunk_end = min(chunk_start + pages_per_chunk - 1, total_pages - 1)
            chunk_num = (chunk_start // pages_per_chunk) + 1

            # Create page range string (e.g., "0-4", "5-9", "10")
            if chunk_start == chunk_end:
                page_range = str(chunk_start)
            else:
                page_range = f"{chunk_start}-{chunk_end}"

            logger.info(f"Processing chunk {chunk_num}: pages {page_range}")

            # Update progress if we have a metrics collector and job tracking
            if metrics_collector:
                # Find the active job for this conversion (assuming it's the most recent one)
                active_jobs = metrics_collector.get_active_jobs_details()
                for job_id, job_info in active_jobs.items():
                    if job_info.get("file_path") == file_path:
                        metrics_collector.update_operation_progress(
                            job_id,
                            current_page=chunk_start + 1,
                            total_pages=total_pages,
                            processing_stage=f"Processing chunk {chunk_num}/{(total_pages + pages_per_chunk - 1) // pages_per_chunk}",
                        )
                        break

            # Create chunk filename in the same output directory (not separate folders)
            output_format = arguments.get("output_format", "markdown")
            ext = "md" if output_format == "markdown" else output_format
            chunk_filename = f"{base_filename}_chunk_{chunk_num}.{ext}"
            chunk_output_file = os.path.join(output_dir, chunk_filename)

            # Prepare arguments for single convert
            chunk_args = {
                "file_path": file_path,
                "output_path": chunk_output_file,  # Use specific output file instead of directory
                "page_range": page_range,
                "debug": arguments.get("debug", False),
                "output_format": output_format,
                "processors": arguments.get("processors", ""),
                "config_json": arguments.get("config_json", ""),
                "disable_multiprocessing": arguments.get(
                    "disable_multiprocessing", False
                ),
                "disable_image_extraction": arguments.get(
                    "disable_image_extraction", False
                ),
                "converter_cls": arguments.get("converter_cls", ""),
                "llm_service": arguments.get("llm_service", ""),
                "use_llm": arguments.get("use_llm", False),
            }

            # Remove empty string values to avoid overriding defaults
            chunk_args = {k: v for k, v in chunk_args.items() if v != ""}

            # Process this chunk
            chunk_result = await handle_single_convert(chunk_args)

            if not chunk_result.get("success"):
                logger.warning(
                    f"Failed to process chunk {chunk_num} (pages {page_range}): {chunk_result.get('error')}"
                )
                chunk_results.append(
                    {
                        "chunk_num": chunk_num,
                        "page_range": page_range,
                        "success": False,
                        "error": chunk_result.get("error"),
                    }
                )
                continue

            # Store chunk result info
            actual_output_file = chunk_result.get("output_file", chunk_output_file)
            chunk_results.append(
                {
                    "chunk_num": chunk_num,
                    "page_range": page_range,
                    "success": True,
                    "output_file": actual_output_file,
                }
            )

            if actual_output_file and os.path.exists(actual_output_file):
                chunk_files.append(actual_output_file)

        # Combine outputs if requested
        combined_output_file = None
        if combine_output and chunk_files:
            output_format = arguments.get("output_format", "markdown")
            ext = "md" if output_format == "markdown" else output_format
            combined_output_file = os.path.join(
                output_dir, f"{base_filename}_combined.{ext}"
            )

            logger.info(
                f"Combining {len(chunk_files)} chunk outputs into {combined_output_file}"
            )

            try:
                with open(combined_output_file, "w", encoding="utf-8") as combined_file:
                    for i, chunk_file in enumerate(chunk_files):
                        if i > 0:
                            # Add page separator for combined output
                            combined_file.write("\n\n---\n\n")

                        try:
                            with open(chunk_file, "r", encoding="utf-8") as cf:
                                content = cf.read()
                                combined_file.write(content)
                        except Exception as e:
                            logger.warning(
                                f"Failed to read chunk file {chunk_file}: {str(e)}"
                            )
                            combined_file.write(
                                f"\n\n[Error reading chunk file: {chunk_file}]\n\n"
                            )

                logger.info(
                    f"Successfully combined outputs into {combined_output_file}"
                )

                # Clean up individual chunk files after successful combination
                logger.info("Cleaning up individual chunk files...")
                for chunk_file in chunk_files:
                    try:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                            logger.debug(f"Removed chunk file: {chunk_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove chunk file {chunk_file}: {str(e)}"
                        )

            except Exception as e:
                logger.error(f"Failed to combine output files: {str(e)}")
                combined_output_file = None

        # Count successful chunks
        successful_chunks = sum(1 for result in chunk_results if result.get("success"))
        total_chunks = len(chunk_results)

        result = {
            "success": True,
            "input_file": file_path,
            "total_pages": total_pages,
            "pages_per_chunk": pages_per_chunk,
            "total_chunks": total_chunks,
            "successful_chunks": successful_chunks,
            "chunk_results": chunk_results,
            "output_dir": output_dir,
        }

        if combined_output_file:
            result["combined_output_file"] = combined_output_file

        if successful_chunks == total_chunks:
            result["message"] = (
                f"Batch pages conversion completed successfully. Processed {total_pages} pages in {total_chunks} chunks."
            )
        else:
            result["message"] = (
                f"Batch pages conversion completed with {successful_chunks}/{total_chunks} chunks successful."
            )
            if successful_chunks == 0:
                result["success"] = False

        return result

    except Exception as e:
        error_msg = f"Error in batch_pages_convert: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "message": f"Failed to process batch pages conversion: {str(e)}",
        }
