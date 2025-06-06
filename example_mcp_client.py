#!/usr/bin/env python3
"""
Example MCP client for Marker MCP Server

This script demonstrates how to use the enhanced Marker MCP server with the full range
of CLI arguments that were previously only available when running marker scripts directly.
"""

import json
import subprocess
import sys
from pathlib import Path


class MarkerMCPClient:
    """Simple MCP client for interacting with Marker MCP server."""
    
    def __init__(self, server_command=None):
        """Initialize the MCP client.
        
        Args:
            server_command: Command to start the MCP server. If None, uses default.
        """
        if server_command is None:
            server_command = [sys.executable, "-m", "src.marker_mcp_server.server"]
        self.server_command = server_command
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            dict: Result from the tool
        """
        # Create MCP request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        # Start server process
        try:
            process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send request
            request_json = json.dumps(request) + "\n"
            stdout, stderr = process.communicate(input=request_json, timeout=30)
            
            # Parse response
            if stdout.strip():
                response = json.loads(stdout.strip())
                return response.get("result", {})
            else:
                return {"success": False, "error": f"No response from server. stderr: {stderr}"}
                
        except subprocess.TimeoutExpired:
            process.kill()
            return {"success": False, "error": "Server timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def example_single_convert_with_args():
    """Example: Convert a single PDF with advanced arguments."""
    print("=== Example: Single PDF conversion with advanced arguments ===")
    
    client = MarkerMCPClient()
    
    # Example arguments that were previously only available via CLI
    arguments = {
        "pdf_path": "/path/to/your/document.pdf",
        "output_path": "/path/to/output/document.md",
        "debug": True,  # Enable debug mode
        "output_format": "markdown",  # Output format
        "page_range": "0-5",  # Convert only first 6 pages
        "use_llm": True,  # Enable LLM processing for higher quality
        "llm_service": "groq",  # Use Groq LLM service
        "disable_image_extraction": False,  # Keep image extraction enabled
        "processors": "marker.processors.text,marker.processors.table",  # Custom processors
        "config_json": "/path/to/custom/config.json"  # Custom configuration
    }
    
    result = client.call_tool("single_convert", arguments)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result


def example_batch_convert_with_chunking():
    """Example: Batch convert with chunking and parallel processing."""
    print("\n=== Example: Batch conversion with chunking ===")
    
    client = MarkerMCPClient()
    
    # Example of batch processing with chunking (useful for large document sets)
    arguments = {
        "folder_path": "/path/to/pdf/folder",
        "output_dir": "/path/to/output/folder",
        "chunk_idx": 0,  # Process first chunk
        "num_chunks": 4,  # Split work into 4 chunks
        "max_files": 100,  # Limit to 100 files
        "workers": 8,  # Use 8 worker processes
        "skip_existing": True,  # Skip already converted files
        "debug": False,
        "output_format": "json",  # Output as JSON
        "use_llm": False,  # Disable LLM for faster processing
        "disable_multiprocessing": False,
        "page_range": "",  # Convert all pages
        "max_tasks_per_worker": 2  # Limit tasks per worker to manage memory
    }
    
    result = client.call_tool("batch_convert", arguments)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result


def example_chunk_convert():
    """Example: Chunk-based folder processing."""
    print("\n=== Example: Chunk-based folder processing ===")
    
    client = MarkerMCPClient()
    
    arguments = {
        "in_folder": "/path/to/large/pdf/collection",
        "out_folder": "/path/to/outputs",
        "chunk_size": 50,  # Process 50 files at a time
        "debug": True,
        "output_format": "html",  # Output as HTML
        "processors": "marker.processors.document_toc",  # Add table of contents
        "use_llm": True,
        "llm_service": "openai"  # Use OpenAI for LLM processing
    }
    
    result = client.call_tool("chunk_convert", arguments)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result


def example_start_api_server():
    """Example: Start the FastAPI server."""
    print("\n=== Example: Start FastAPI server ===")
    
    client = MarkerMCPClient()
    
    arguments = {
        "host": "0.0.0.0",  # Listen on all interfaces
        "port": 8080  # Custom port
    }
    
    result = client.call_tool("start_server", arguments)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    return result


def main():
    """Run example usage scenarios."""
    print("Marker MCP Server - Enhanced Argument Support Examples")
    print("=" * 60)
    print()
    print("This script demonstrates how to use the enhanced MCP server")
    print("with all the CLI arguments that were previously only available")
    print("when running marker scripts directly.")
    print()
    
    # Note: These examples use placeholder paths
    print("NOTE: The examples below use placeholder file paths.")
    print("Update the paths to actual files to test with real data.")
    print()
    
    try:
        # Run examples (with placeholder data)
        example_single_convert_with_args()
        example_batch_convert_with_chunking()
        example_chunk_convert()
        example_start_api_server()
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print()
    print("Key improvements in the enhanced MCP server:")
    print("• Full CLI argument compatibility")
    print("• Debug mode support")
    print("• Custom output formats (markdown, json, html)")
    print("• Page range selection")
    print("• LLM integration options")
    print("• Chunking and parallel processing")
    print("• Custom processor selection")
    print("• Configuration file support")
    print("• Memory management options")


if __name__ == "__main__":
    main()
