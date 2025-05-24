#!/usr/bin/env python3
"""
Marker MCP Server - MCP-only mode for Claude Desktop integration
"""
import sys
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp.server.stdio import stdio_server
    from mcp.server import Server
    from mcp import types
    from pydantic.networks import AnyUrl
    from mcp.server.models import InitializationOptions
    from mcp.server.lowlevel.server import NotificationOptions
except ImportError as e:
    print(f"Error importing MCP: {e}", file=sys.stderr)
    sys.exit(1)

# Try to import project modules with fallbacks
try:
    from unified_monitoring import UnifiedMonitoringSystem
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    print("Warning: unified_monitoring not available", file=sys.stderr)

try:
    from config_fixes.optimized_config import load_config, save_config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("Warning: optimized_config not available", file=sys.stderr)

try:
    from unified_conversion import UnifiedConversionSystem
    HAS_CONVERSION = True
except ImportError:
    HAS_CONVERSION = False
    print("Warning: unified_conversion not available", file=sys.stderr)

# Initialize the MCP server
server = Server("marker-server")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri=AnyUrl("monitor://status"),
            name="Monitor Status",
            description="Get current monitoring status summary",
            mimeType="text/plain"
        ),
        types.Resource(
            uri=AnyUrl("config://settings"), 
            name="Configuration Settings",
            description="Get current configuration settings",
            mimeType="application/json"
        ),
        types.Resource(
            uri=AnyUrl("files://uploads"),
            name="Upload Files",
            description="List available files in uploads directory", 
            mimeType="application/json"
        )
    ]
@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Handle resource read requests"""
    uri_str = str(uri)
    print(f"DEBUG: Reading resource: {uri_str} (type: {type(uri)})", file=sys.stderr)
    
    if uri_str == "monitor://status":
        print("DEBUG: Matched monitor://status", file=sys.stderr)
        return await get_monitor_status()
    elif uri_str == "config://settings":
        print("DEBUG: Matched config://settings", file=sys.stderr)
        result = await get_config_settings()
        return json.dumps(result, indent=2)
    elif uri_str == "files://uploads":
        print("DEBUG: Matched files://uploads", file=sys.stderr)
        return await list_upload_files()
    else:
        print(f"DEBUG: No match for resource: {uri_str}", file=sys.stderr)
        raise ValueError(f"Unknown resource: {uri_str}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="start_monitor",
            description="Start real-time monitoring",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="stop_monitor", 
            description="Stop real-time monitoring",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="plot_monitor",
            description="Generate performance plot",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to plot",
                        "default": 7
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="update_config",
            description="Update configuration settings",
            inputSchema={
                "type": "object", 
                "properties": {
                    "config_json": {
                        "type": "string",
                        "description": "JSON string with configuration updates"
                    }
                },
                "required": ["config_json"]
            }
        ),
        types.Tool(
            name="convert_file",
            description="Convert a file to markdown", 
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to convert"
                    },
                    "output_path": {
                        "type": "string", 
                        "description": "Optional output path for converted file"
                    }
                },
                "required": ["file_path"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    if name == "start_monitor":
        result = await start_monitor()
    elif name == "stop_monitor":
        result = await stop_monitor()
    elif name == "plot_monitor":
        days = arguments.get("days", 7)
        result = await plot_monitor(days)
    elif name == "update_config":
        config_json = arguments.get("config_json", "{}")
        result = await update_config_tool(config_json)
    elif name == "convert_file":
        file_path = arguments.get("file_path")
        if file_path is None:
            result = "Error: file_path argument is required for convert_file tool."
        else:
            output_path = arguments.get("output_path")
            result = await convert_file_tool(file_path, output_path)
    else:
        raise ValueError(f"Unknown tool: {name}")
    
    return [types.TextContent(type="text", text=result)]

# Implementation functions
async def get_monitor_status() -> str:
    """Get current monitoring status summary"""
    if not HAS_MONITORING:
        return "Monitoring system not available"
    
    try:
        # Ensure monitoring directory exists
        monitoring_dir = project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Change to project directory for monitoring
        old_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            monitor = UnifiedMonitoringSystem()
            # Capture status summary
            from io import StringIO
            import sys as sys_module
            buf = StringIO()
            old_stdout = sys_module.stdout
            try:
                sys_module.stdout = buf
                monitor.print_status_summary()
                return buf.getvalue()
            finally:
                sys_module.stdout = old_stdout
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return f"Error getting status: {str(e)}"

async def start_monitor() -> str:
    """Start real-time monitoring"""
    if not HAS_MONITORING:
        return "Monitoring system not available"
    
    try:
        # Ensure monitoring directory exists
        monitoring_dir = project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Change to project directory for monitoring
        old_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            monitor = UnifiedMonitoringSystem()
            monitor.start_realtime_monitoring()
            return "Monitoring started successfully"
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return f"Error starting monitoring: {str(e)}"

async def stop_monitor() -> str:
    """Stop real-time monitoring"""
    if not HAS_MONITORING:
        return "Monitoring system not available"
    
    try:
        # Change to project directory for monitoring
        old_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            monitor = UnifiedMonitoringSystem()
            monitor.stop_realtime_monitoring()
            return "Monitoring stopped successfully"
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return f"Error stopping monitoring: {str(e)}"

async def plot_monitor(days: int = 7) -> str:
    """Generate performance plot"""
    if not HAS_MONITORING:
        return "Monitoring system not available"
    
    try:
        # Ensure monitoring directory exists
        monitoring_dir = project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Change to project directory for monitoring
        old_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            monitor = UnifiedMonitoringSystem()
            path = f"monitoring/plot_{days}d.png"
            monitor.generate_performance_plot(days, save_path=path)
            return f"Plot generated: {path}"
        finally:
            os.chdir(old_cwd)
    except Exception as e:
        return f"Error generating plot: {str(e)}"

async def get_config_settings() -> dict:
    """Get current configuration settings"""
    if not HAS_CONFIG:
        return {"error": "Configuration system not available"}
        
    try:
        return load_config()
    except Exception as e:
        return {"error": str(e)}

async def update_config_tool(config_json: str) -> str:
    """Update configuration settings"""
    if not HAS_CONFIG:
        return "Configuration system not available"
        
    try:
        config = json.loads(config_json)
        save_config(config)
        return "Configuration updated successfully"
    except Exception as e:
        return f"Error updating config: {str(e)}"

async def convert_file_tool(file_path: str, output_path: Optional[str] = None) -> str:
    """Convert a file to markdown"""
    if not HAS_CONVERSION:
        return "Conversion system not available"
    
    try:
        # Use absolute paths based on project root
        input_file = Path(file_path)
        if not input_file.is_absolute():
            input_file = project_root / file_path
            
        if not input_file.exists():
            return f"Error: File not found: {input_file}"
        
        if output_path is None:
            output_path = str(project_root / f"conversion_results/{input_file.stem}.md")
        elif not Path(output_path).is_absolute():
            output_path = str(project_root / output_path)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Change to project directory for conversion
        old_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            converter = UnifiedConversionSystem()
            # Use the correct method for single PDF conversion
            result = converter.convert_single_pdf(str(input_file), output_path)
            
            if result and result.success and Path(output_path).exists():
                return f"File converted successfully: {output_path}"
            elif result and result.error_message:
                return f"Conversion failed: {result.error_message}"
            else:
                return "Conversion failed"
        finally:
            os.chdir(old_cwd)
            
    except Exception as e:
        return f"Error converting file: {str(e)}"

async def list_upload_files() -> str:
    """List available files in uploads directory"""
    try:
        uploads_dir = project_root / "uploads"
        print(f"DEBUG: Looking for uploads at: {uploads_dir}", file=sys.stderr)
        print(f"DEBUG: uploads_dir.exists(): {uploads_dir.exists()}", file=sys.stderr)
        
        if not uploads_dir.exists():
            return f"No uploads directory found at {uploads_dir}"
        
        files = []
        for file_path in uploads_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "path": str(file_path)
                })
        
        print(f"DEBUG: Found {len(files)} files", file=sys.stderr)
        return json.dumps(files, indent=2)
    except Exception as e:
        print(f"DEBUG: Error in list_upload_files: {e}", file=sys.stderr)
        return f"Error listing files: {str(e)}"

async def run_server():
    """Run the MCP server"""
    print("Starting Marker MCP Server...", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="marker-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

def main():
    """Entry point for the MCP server"""
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
