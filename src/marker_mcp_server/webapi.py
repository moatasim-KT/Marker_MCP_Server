#!/usr/bin/env python3
"""Web API Bridge for Marker MCP Server.

This module creates a FastAPI bridge that exposes MCP tools as REST endpoints
for the web interface.
"""
import asyncio
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .tools import (
    handle_batch_convert,
    handle_batch_pages_convert,
    handle_chunk_convert,
    handle_single_convert,
)
from .monitoring import get_metrics_collector
from .config import Config

logger = logging.getLogger(__name__)

# Pydantic models for API requests
class ConvertSingleRequest(BaseModel):
    output_format: str = "markdown"
    use_llm: bool = False
    max_pages: Optional[int] = None
    page_range: Optional[str] = None
    debug: bool = False

class ConvertBatchRequest(BaseModel):
    output_format: str = "markdown"
    use_llm: bool = False
    debug: bool = False

class ConvertChunkRequest(BaseModel):
    pages_per_chunk: int = 5
    combine_output: bool = True
    output_format: str = "markdown"
    use_llm: bool = False
    debug: bool = False

# Global variables for job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
job_history: List[Dict[str, Any]] = []
connected_websockets: List[WebSocket] = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove disconnected connections
                self.disconnect(connection)

manager = ConnectionManager()

def create_web_app() -> FastAPI:
    """Create and configure the FastAPI web application."""
    app = FastAPI(
        title="Marker MCP Server Web Interface",
        description="Web interface for PDF conversion and monitoring",
        version="1.0.0"
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files for React build
    webui_build_path = Path(__file__).parent / "webui" / "build"
    if webui_build_path.exists():
        app.mount("/static", StaticFiles(directory=str(webui_build_path / "static")), name="static")

    @app.get("/")
    async def read_root():
        """Serve the React app."""
        if webui_build_path.exists():
            return FileResponse(str(webui_build_path / "index.html"))
        return {"message": "Marker MCP Server Web API", "status": "running"}

    @app.get("/api/health")
    async def get_health():
        """Get system health status."""
        try:
            metrics_collector = get_metrics_collector()
            if not metrics_collector:
                return {"status": "error", "message": "Monitoring not initialized"}
            
            health = metrics_collector.get_system_health()
            return {
                "status": health.status,
                "memory_status": health.memory_status,
                "processing_status": health.processing_status,
                "active_jobs": health.active_jobs,
                "queue_size": health.queue_size,
                "alerts": health.alerts,
                "timestamp": health.timestamp
            }
        except Exception as e:
            logger.error(f"Error getting health: {e}")
            return {"status": "error", "message": str(e)}

    @app.get("/api/metrics")
    async def get_metrics(hours: int = 1):
        """Get performance metrics summary."""
        try:
            metrics_collector = get_metrics_collector()
            if not metrics_collector:
                return {"error": "Monitoring not initialized"}
            
            summary = metrics_collector.get_metrics_summary(hours)
            return {"success": True, "metrics": summary}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"success": False, "error": str(e)}

    @app.get("/api/jobs")
    async def get_jobs():
        """Get active jobs and job history."""
        return {
            "active_jobs": active_jobs,
            "job_history": job_history[-50:]  # Last 50 jobs
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Get status of a specific job."""
        if job_id in active_jobs:
            return active_jobs[job_id]
        
        # Check job history
        for job in job_history:
            if job["id"] == job_id:
                return job
        
        raise HTTPException(status_code=404, detail="Job not found")

    @app.post("/api/convert/single")
    async def convert_single_file(
        file: UploadFile = File(...),
        config: str = Form(default='{}')
    ):
        """Convert a single PDF file."""
        job_id = str(uuid.uuid4())
        
        try:
            # Parse configuration
            request_config = ConvertSingleRequest(**json.loads(config))
            
            # Save uploaded file
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, file.filename)
            
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create job entry
            job_data = {
                "id": job_id,
                "type": "single_convert",
                "status": "processing",
                "filename": file.filename,
                "started_at": datetime.now().isoformat(),
                "progress": 0
            }
            active_jobs[job_id] = job_data
            
            # Broadcast job start
            await manager.broadcast({
                "type": "job_started",
                "job": job_data
            })
            
            # Prepare arguments
            arguments = {
                "file_path": input_path,
                "output_dir": temp_dir,
                "output_format": request_config.output_format,
                "use_llm": request_config.use_llm,
                "debug": request_config.debug
            }
            
            if request_config.max_pages:
                arguments["max_pages"] = request_config.max_pages
            if request_config.page_range:
                arguments["page_range"] = request_config.page_range
            
            # Process the file
            result = await handle_single_convert(arguments)
            
            # Update job status
            job_data.update({
                "status": "completed" if result.get("success") else "failed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
                "progress": 100
            })
            
            # Move to history and remove from active
            job_history.append(job_data.copy())
            active_jobs.pop(job_id, None)
            
            # Broadcast completion
            await manager.broadcast({
                "type": "job_completed",
                "job": job_data
            })
            
            return {
                "job_id": job_id,
                "status": job_data["status"],
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in single convert: {e}")
            
            # Update job with error
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })
                job_history.append(active_jobs[job_id].copy())
                active_jobs.pop(job_id, None)
            
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/convert/batch")
    async def convert_batch_files(
        files: List[UploadFile] = File(...),
        config: str = Form(default='{}')
    ):
        """Convert multiple PDF files."""
        job_id = str(uuid.uuid4())
        
        try:
            request_config = ConvertBatchRequest(**json.loads(config))
            
            # Save uploaded files
            temp_dir = tempfile.mkdtemp()
            input_folder = os.path.join(temp_dir, "input")
            os.makedirs(input_folder)
            
            filenames = []
            for file in files:
                file_path = os.path.join(input_folder, file.filename)
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                filenames.append(file.filename)
            
            # Create job entry
            job_data = {
                "id": job_id,
                "type": "batch_convert",
                "status": "processing",
                "filenames": filenames,
                "total_files": len(files),
                "started_at": datetime.now().isoformat(),
                "progress": 0
            }
            active_jobs[job_id] = job_data
            
            await manager.broadcast({
                "type": "job_started",
                "job": job_data
            })
            
            # Process batch
            arguments = {
                "input_folder": input_folder,
                "output_dir": os.path.join(temp_dir, "output"),
                "output_format": request_config.output_format,
                "use_llm": request_config.use_llm,
                "debug": request_config.debug
            }
            
            result = await handle_batch_convert(arguments)
            
            # Update job status
            job_data.update({
                "status": "completed" if result.get("success") else "failed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
                "progress": 100
            })
            
            job_history.append(job_data.copy())
            active_jobs.pop(job_id, None)
            
            await manager.broadcast({
                "type": "job_completed",
                "job": job_data
            })
            
            return {
                "job_id": job_id,
                "status": job_data["status"],
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in batch convert: {e}")
            
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })
                job_history.append(active_jobs[job_id].copy())
                active_jobs.pop(job_id, None)
            
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/convert/chunk")
    async def convert_chunk_file(
        file: UploadFile = File(...),
        config: str = Form(default='{}')
    ):
        """Convert a PDF file using chunked processing."""
        job_id = str(uuid.uuid4())
        
        try:
            request_config = ConvertChunkRequest(**json.loads(config))
            
            # Save uploaded file
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, file.filename)
            
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Create job entry
            job_data = {
                "id": job_id,
                "type": "chunk_convert",
                "status": "processing",
                "filename": file.filename,
                "started_at": datetime.now().isoformat(),
                "progress": 0
            }
            active_jobs[job_id] = job_data
            
            await manager.broadcast({
                "type": "job_started",
                "job": job_data
            })
            
            # Process chunks
            arguments = {
                "file_path": input_path,
                "output_dir": temp_dir,
                "pages_per_chunk": request_config.pages_per_chunk,
                "combine_output": request_config.combine_output,
                "output_format": request_config.output_format,
                "use_llm": request_config.use_llm,
                "debug": request_config.debug
            }
            
            result = await handle_batch_pages_convert(arguments)
            
            # Update job status
            job_data.update({
                "status": "completed" if result.get("success") else "failed",
                "completed_at": datetime.now().isoformat(),
                "result": result,
                "progress": 100
            })
            
            job_history.append(job_data.copy())
            active_jobs.pop(job_id, None)
            
            await manager.broadcast({
                "type": "job_completed",
                "job": job_data
            })
            
            return {
                "job_id": job_id,
                "status": job_data["status"],
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error in chunk convert: {e}")
            
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })
                job_history.append(active_jobs[job_id].copy())
                active_jobs.pop(job_id, None)
            
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                # Echo back for now, can add more functionality later
                await websocket.send_text(f"Echo: {data}")
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    return app

def start_web_server(host: str = "127.0.0.1", port: int = 8080, reload: bool = False):
    """Start the web server."""
    app = create_web_app()
    uvicorn.run(app, host=host, port=port, reload=reload)

if __name__ == "__main__":
    start_web_server()
