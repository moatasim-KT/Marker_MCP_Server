"""
Dashboard module for Marker MCP Server.

This module provides a web-based dashboard for monitoring the Marker MCP server
in real-time, displaying metrics such as CPU/GPU usage, memory consumption,
and conversion progress.
"""

import asyncio
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import monitoring functions and types
MONITORING_AVAILABLE = False
_monitoring_get_metrics_collector = None

try:
    from .monitoring import get_metrics_collector as _monitoring_get_metrics_collector

    MONITORING_AVAILABLE = True
except ImportError:
    pass  # Use fallback


def get_metrics_collector():
    """Get metrics collector if available, otherwise return None."""
    if MONITORING_AVAILABLE and _monitoring_get_metrics_collector:
        return _monitoring_get_metrics_collector()
    return None


# Dashboard SystemHealth Pydantic model (separate from monitoring SystemHealth)
class DashboardSystemHealth(BaseModel):
    status: str = "unknown"
    timestamp: str = datetime.datetime.utcnow().isoformat()
    memory_status: str = "unknown"
    processing_status: str = "inactive"
    alerts: List[str] = ["Monitoring module not available"]
    active_jobs: int = 0
    queue_size: int = 0

    @classmethod
    def create_default(cls) -> "DashboardSystemHealth":
        """Create a default health status."""
        return cls()


# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Set up paths
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

logger = logging.getLogger("marker-mcp-dashboard")

# Create FastAPI app
app = FastAPI()

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"New WebSocket connection. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}"
            )

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# API endpoints
@app.get("/api/health")
async def get_health() -> Dict[str, Any]:
    """Get current system health status."""
    collector = get_metrics_collector()
    if collector and hasattr(collector, "get_system_health"):
        health = collector.get_system_health()
        if health:
            # Handle dataclass SystemHealth from monitoring module
            if hasattr(health, "__dataclass_fields__"):
                # It's a dataclass, convert to dict using asdict or manual conversion
                try:
                    from dataclasses import asdict

                    return asdict(health)
                except ImportError:
                    # Fallback to manual conversion
                    return {
                        "timestamp": getattr(health, "timestamp", time.time()),
                        "status": getattr(health, "status", "unknown"),
                        "memory_status": getattr(health, "memory_status", "unknown"),
                        "processing_status": getattr(
                            health, "processing_status", "inactive"
                        ),
                        "alerts": getattr(health, "alerts", []),
                        "active_jobs": getattr(health, "active_jobs", 0),
                        "queue_size": getattr(health, "queue_size", 0),
                    }
            # Handle Pydantic models
            elif hasattr(health, "dict") and callable(getattr(health, "dict")):
                return health.dict()  # type: ignore
            else:
                # Fallback to vars() and convert to regular dict
                return dict(vars(health))

    # Return a default health status
    return DashboardSystemHealth.create_default().dict()


# API endpoint for the dashboard
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}


# Test endpoint to verify dashboard is working
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify dashboard is accessible."""
    return {
        "message": "Dashboard is working!",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "ok",
    }


# Background task to send metrics updates
async def metrics_broadcaster():
    last_error_time = 0
    error_count = 0
    max_errors_before_backoff = 5

    while True:
        try:
            # Get metrics from the collector
            metrics = {}
            collector = get_metrics_collector()

            try:
                if collector:
                    metrics = collector.get_metrics()
                else:
                    # Fallback metrics if collector is not available
                    import psutil

                    process = psutil.Process()
                    memory_info = process.memory_info()

                    metrics = {
                        "cpu_percent": process.cpu_percent(interval=0.1),
                        "memory_percent": process.memory_percent(),
                        "memory_mb": memory_info.rss / (1024 * 1024),  # Convert to MB
                        "memory_available_mb": psutil.virtual_memory().available
                        / (1024 * 1024),
                        "memory_total_mb": psutil.virtual_memory().total
                        / (1024 * 1024),
                        "gpu_memory_percent": 0,  # Will be 0 if no GPU
                        "gpu_memory_mb": 0,  # Will be 0 if no GPU
                        "active_jobs": 0,
                        "queue_size": 0,
                        "process_count": len(psutil.pids()),
                        "timestamp": time.time(),
                    }

                # Reset error count on successful metrics collection
                if error_count > 0:
                    logger.info("Metrics collection recovered successfully")
                    error_count = 0

                # Get detailed active jobs information
                active_jobs_details = {}
                if collector:
                    active_jobs_details = collector.get_active_jobs_details()

                # Broadcast to all connected clients with proper structure
                dashboard_data = {
                    "type": "metrics_update",
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "health": {
                        "status": "healthy",
                        "cpu_percent": metrics.get("cpu_percent", 0),
                        "memory_percent": metrics.get("memory_percent", 0),
                        "gpu_memory_percent": metrics.get("gpu_memory_percent", 0),
                        "gpu_type": metrics.get("gpu_type", "None"),
                        "gpu_device_name": metrics.get("gpu_device_name", "No GPU"),
                        "memory_status": "normal",
                        "processing_status": "active"
                        if metrics.get("active_jobs", 0) > 0
                        else "idle",
                        "active_jobs": metrics.get("active_jobs", 0),
                        "queue_size": metrics.get("queue_size", 0),
                        "alerts": [],
                    },
                    "active_jobs": active_jobs_details,
                    "metrics": metrics,
                }
                await manager.broadcast(json.dumps(dashboard_data))

            except Exception as e:
                current_time = time.time()
                error_count += 1

                # Log full error only once per minute to avoid log spam
                if current_time - last_error_time > 60:
                    logger.error(
                        f"Error collecting metrics (error count: {error_count}): {e}",
                        exc_info=True,
                    )
                    last_error_time = current_time

                # If we've had multiple errors in a row, increase the delay
                if error_count > max_errors_before_backoff:
                    backoff_time = min(
                        60, 2 ** (error_count - max_errors_before_backoff)
                    )  # Exponential backoff, max 60s
                    logger.warning(
                        f"Multiple errors detected, backing off for {backoff_time} seconds..."
                    )
                    await asyncio.sleep(backoff_time)
                else:
                    await asyncio.sleep(1)  # Normal update interval

                continue

        except Exception as e:
            logger.error(f"Unexpected error in metrics broadcaster: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait a bit before retrying after unexpected errors
            continue

        await asyncio.sleep(1)  # Normal update interval


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Start the metrics broadcaster when the app starts")
    asyncio.create_task(metrics_broadcaster())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down dashboard")
