#!/usr/bin/env python3
"""
Integrated Server Launcher for Marker MCP Server
Starts both the MCP server and the web interface.
"""
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.marker_mcp_server.webapi import create_web_app
from src.marker_mcp_server.monitoring import init_monitoring

logger = logging.getLogger(__name__)

def start_react_dev_server():
    """Start the React development server."""
    webui_path = Path(__file__).parent / "webui"
    if not webui_path.exists():
        logger.error(f"WebUI directory not found: {webui_path}")
        return
    
    try:
        logger.info("Starting React development server...")
        env = os.environ.copy()
        env["BROWSER"] = "none"  # Don't auto-open browser
        env["PORT"] = "3000"
        
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=str(webui_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor the process
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[React] {line.strip()}")
                if "compiled successfully" in line.lower():
                    logger.info("React development server started successfully")
                    break
        
        # Keep process running
        process.wait()
        
    except Exception as e:
        logger.error(f"Failed to start React server: {e}")

def start_fastapi_server():
    """Start the FastAPI web server."""
    try:
        logger.info("Starting FastAPI web server...")
        
        # Initialize monitoring
        init_monitoring()
        
        # Create and run the FastAPI app
        app = create_web_app()
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start FastAPI server: {e}")

def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Marker MCP Server with Web Interface...")
    
    # Start React dev server in a separate thread
    react_thread = threading.Thread(target=start_react_dev_server, daemon=True)
    react_thread.start()
    
    # Wait a moment for React to start
    time.sleep(2)
    
    # Start FastAPI server (blocking)
    start_fastapi_server()

if __name__ == "__main__":
    main()
