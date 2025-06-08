"""Performance monitoring and metrics collection for Marker MCP Server.

This module provides comprehensive monitoring capabilities including:
- Resource usage tracking (memory, CPU, disk)
- Performance metrics collection
- Alert management
- Metrics persistence and reporting
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics at a point in time."""

    timestamp: float
    memory_mb: float
    memory_percent: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    process_count: int = 1


@dataclass
class PerformanceMetrics:
    """Performance metrics for processing operations."""

    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    pages_processed: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_peak_mb: Optional[float] = None
    cpu_time_seconds: Optional[float] = None
    # Enhanced progress tracking
    total_pages: Optional[int] = None
    current_page: Optional[int] = None
    processing_stage: Optional[str] = None
    progress_percent: Optional[float] = None
    file_path: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""

    timestamp: float
    status: str  # "healthy", "warning", "critical"
    memory_status: str
    processing_status: str
    alerts: List[str]
    active_jobs: int
    queue_size: int


class MetricsCollector:
    """Collects and manages system and performance metrics."""

    def __init__(self, config: Any):
        self.config = config
        # Access monitoring config directly from the config object
        self.monitoring_config = config.monitoring
        self.resource_limits = config.resource_limits
        self.paths_config = config.paths

        self._metrics_history: List[ResourceMetrics] = []
        self._performance_history: List[PerformanceMetrics] = []
        self._alerts: List[str] = []
        self._active_jobs: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.Lock()
        self._collector_task: Optional[asyncio.Task] = None
        self._running = False

        # Ensure metrics directory exists
        Path(self.paths_config.metrics_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths_config.logs_dir).mkdir(parents=True, exist_ok=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.

        Returns:
            dict: Dictionary containing current system metrics
        """
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        metrics = {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_percent": process.memory_percent(),
            "memory_mb": memory_info.rss / (1024 * 1024),  # Convert to MB
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
            "active_jobs": len(self._active_jobs),
            "queue_size": 0,  # Update this if you have a queue system
            "gpu_memory_percent": 0,  # Will be 0 if no GPU
            "gpu_memory_mb": 0,  # Will be 0 if no GPU
            "disk_usage_mb": 0,  # Update this if you track disk usage
            "process_count": len(psutil.pids()),
            "timestamp": time.time(),
        }

        # Add GPU metrics if available
        if HAS_TORCH:
            try:
                # Check for CUDA first
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (
                        1024 * 1024
                    )
                    metrics.update(
                        {
                            "gpu_memory_mb": gpu_mem,
                            "gpu_memory_percent": (gpu_mem / gpu_mem_total) * 100
                            if gpu_mem_total > 0
                            else 0,
                            "gpu_type": "CUDA",
                            "gpu_device_name": torch.cuda.get_device_name(0)
                            if torch.cuda.device_count() > 0
                            else "Unknown",
                        }
                    )
                # Check for MPS (Apple Silicon)
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    # MPS memory tracking
                    gpu_mem = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
                    # MPS doesn't have a direct way to get total memory, so we estimate based on system memory
                    # Apple Silicon typically shares memory between CPU and GPU
                    total_system_memory = psutil.virtual_memory().total / (1024 * 1024)
                    # Assume GPU can use up to 75% of system memory (conservative estimate)
                    gpu_mem_total = total_system_memory * 0.75
                    gpu_percent = (
                        (gpu_mem / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
                    )

                    metrics.update(
                        {
                            "gpu_memory_mb": gpu_mem,
                            "gpu_memory_percent": min(gpu_percent, 100),  # Cap at 100%
                            "gpu_type": "MPS",
                            "gpu_device_name": "Apple Silicon GPU",
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")

        return metrics

    async def start(self):
        """Start the metrics collection background task."""
        if self._running:
            return

        self._running = True
        if self.monitoring_config.enable_metrics:
            self._collector_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Metrics collection started")

    async def stop(self):
        """Stop the metrics collection background task."""
        self._running = False
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
            logger.info("Metrics collection stopped")

    async def _collect_metrics_loop(self):
        """Background task to collect metrics at regular intervals."""
        while self._running:
            try:
                await self._collect_resource_metrics()
                await asyncio.sleep(self.monitoring_config.metrics_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Brief delay before retrying

    async def _collect_resource_metrics(self):
        """Collect current resource usage metrics."""
        if not HAS_PSUTIL or psutil is None:
            return

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()

            # GPU metrics if available
            gpu_memory_mb = None
            gpu_memory_percent = None
            if HAS_TORCH and torch is not None:
                try:
                    # Check for CUDA first
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        gpu_memory_total = torch.cuda.get_device_properties(
                            0
                        ).total_memory / (1024 * 1024)
                        gpu_memory_percent = (
                            (gpu_memory_mb / gpu_memory_total) * 100
                            if gpu_memory_total > 0
                            else 0
                        )
                    # Check for MPS (Apple Silicon)
                    elif (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        gpu_memory_mb = torch.mps.current_allocated_memory() / (
                            1024 * 1024
                        )
                        # Estimate total GPU memory as 75% of system memory for Apple Silicon
                        total_system_memory = psutil.virtual_memory().total / (
                            1024 * 1024
                        )
                        gpu_memory_total = total_system_memory * 0.75
                        gpu_memory_percent = (
                            min((gpu_memory_mb / gpu_memory_total) * 100, 100)
                            if gpu_memory_total > 0
                            else 0
                        )
                except Exception:
                    pass

            # Disk usage for cache directory
            disk_usage_mb = None
            try:
                cache_path = Path(self.paths_config.cache_dir)
                if cache_path.exists():
                    disk_usage_mb = sum(
                        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
                    ) / (1024 * 1024)
            except Exception:
                pass

            metrics = ResourceMetrics(
                timestamp=time.time(),
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_memory_percent=gpu_memory_percent,
                disk_usage_mb=disk_usage_mb,
                process_count=len(self._active_jobs),
            )

            with self._lock:
                self._metrics_history.append(metrics)
                # Keep only last 1000 entries
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-1000:]

            # Check for alerts
            await self._check_resource_alerts(metrics)

            if self.monitoring_config.log_memory_usage:
                logger.debug(f"Memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")

        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")

    async def _check_resource_alerts(self, metrics: ResourceMetrics):
        """Check resource metrics against thresholds and generate alerts."""
        alerts = []

        # Memory alerts
        if (
            metrics.memory_percent
            > self.monitoring_config.alert_memory_threshold_percent
        ):
            alert = f"High memory usage: {metrics.memory_percent:.1f}% (limit: {self.monitoring_config.alert_memory_threshold_percent}%)"
            alerts.append(alert)

        # Check against absolute memory limit
        if metrics.memory_mb > self.resource_limits.max_memory_usage_mb:
            alert = f"Memory usage exceeded limit: {metrics.memory_mb:.1f}MB (limit: {self.resource_limits.max_memory_usage_mb}MB)"
            alerts.append(alert)

        # Check for long-running jobs
        current_time = time.time()
        for job_id, job_metrics in self._active_jobs.items():
            duration = current_time - job_metrics.start_time
            if (
                duration
                > self.monitoring_config.alert_processing_time_threshold_seconds
            ):
                alert = f"Long-running job detected: {job_id} ({duration:.1f}s)"
                alerts.append(alert)

        if alerts:
            with self._lock:
                self._alerts.extend(alerts)
                # Keep only recent alerts
                cutoff_time = current_time - 3600  # 1 hour
                self._alerts = [a for a in self._alerts if cutoff_time < current_time]

            for alert in alerts:
                logger.warning(f"ALERT: {alert}")

    def start_operation(self, operation: str, file_path: Optional[str] = None) -> str:
        """Start tracking a new operation."""
        job_id = f"{operation}_{int(time.time() * 1000)}"

        # Check if we're at the concurrent job limit
        if len(self._active_jobs) >= self.resource_limits.max_concurrent_jobs:
            raise ValueError(
                f"Maximum concurrent jobs limit reached ({self.resource_limits.max_concurrent_jobs})"
            )

        file_size_mb = None
        if file_path and os.path.exists(file_path):
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > self.resource_limits.max_file_size_mb:
                    raise ValueError(
                        f"File size {file_size_mb:.1f}MB exceeds limit of {self.resource_limits.max_file_size_mb}MB"
                    )
            except ValueError:
                # Re-raise ValueError for file size limits
                raise
            except (OSError, IOError) as e:
                logger.warning(f"Could not get file size for {file_path}: {e}")

        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            file_size_mb=file_size_mb,
            file_path=file_path,
        )

        with self._lock:
            self._active_jobs[job_id] = metrics

        logger.info(f"Started operation: {job_id} ({operation})")
        return job_id

    def update_operation_progress(
        self,
        job_id: str,
        current_page: Optional[int] = None,
        total_pages: Optional[int] = None,
        processing_stage: Optional[str] = None,
    ):
        """Update progress information for an active operation."""
        with self._lock:
            if job_id not in self._active_jobs:
                logger.warning(
                    f"Attempted to update progress for unknown job: {job_id}"
                )
                return

            metrics = self._active_jobs[job_id]

            if current_page is not None:
                metrics.current_page = current_page
            if total_pages is not None:
                metrics.total_pages = total_pages
            if processing_stage is not None:
                metrics.processing_stage = processing_stage

            # Calculate progress percentage
            if (
                metrics.current_page is not None
                and metrics.total_pages is not None
                and metrics.total_pages > 0
            ):
                metrics.progress_percent = (
                    metrics.current_page / metrics.total_pages
                ) * 100

            logger.debug(
                f"Updated progress for {job_id}: page {current_page}/{total_pages}, stage: {processing_stage}"
            )

    def get_active_jobs_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all active jobs."""
        with self._lock:
            jobs_details = {}
            for job_id, metrics in self._active_jobs.items():
                jobs_details[job_id] = {
                    "operation": metrics.operation,
                    "started_at": metrics.start_time,
                    "file_path": metrics.file_path,
                    "file_size_mb": metrics.file_size_mb,
                    "current_page": metrics.current_page,
                    "total_pages": metrics.total_pages,
                    "processing_stage": metrics.processing_stage,
                    "progress_percent": metrics.progress_percent or 0,
                    "duration_so_far": time.time() - metrics.start_time,
                }
            return jobs_details

    def end_operation(
        self,
        job_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        pages_processed: Optional[int] = None,
    ):
        """End tracking of an operation."""
        with self._lock:
            if job_id not in self._active_jobs:
                logger.warning(f"Attempted to end unknown job: {job_id}")
                return

            metrics = self._active_jobs.pop(job_id)

        metrics.end_time = time.time()
        metrics.duration_seconds = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.error_message = error_message
        metrics.pages_processed = pages_processed

        # Collect peak memory usage if available
        if HAS_PSUTIL and psutil is not None:
            try:
                process = psutil.Process()
                metrics.memory_peak_mb = process.memory_info().rss / (1024 * 1024)
                metrics.cpu_time_seconds = sum(
                    process.cpu_times()[:2]
                )  # user + system time
            except Exception:
                pass

        with self._lock:
            self._performance_history.append(metrics)
            # Keep only last 500 entries
            if len(self._performance_history) > 500:
                self._performance_history = self._performance_history[-500:]

        if self.monitoring_config.log_performance:
            status = "SUCCESS" if success else "FAILED"
            logger.info(
                f"Operation completed: {job_id} ({status}) - "
                f"Duration: {metrics.duration_seconds:.2f}s"
                + (f", Pages: {pages_processed}" if pages_processed else "")
                + (f", Error: {error_message}" if error_message else "")
            )

    @asynccontextmanager
    async def track_operation(self, operation: str, file_path: Optional[str] = None):
        """Context manager for tracking operations."""
        job_id = self.start_operation(operation, file_path)
        try:
            yield job_id
            self.end_operation(job_id, success=True)
        except Exception as e:
            self.end_operation(job_id, success=False, error_message=str(e))
            raise

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        current_time = time.time()

        with self._lock:
            recent_metrics = [
                m for m in self._metrics_history if current_time - m.timestamp < 300
            ]  # Last 5 minutes
            recent_alerts = [
                a for a in self._alerts if current_time - time.time() < 3600
            ]  # Last hour
            active_jobs = len(self._active_jobs)

        # Determine status
        status = "healthy"
        memory_status = "ok"
        processing_status = "ok"

        if recent_metrics:
            latest = recent_metrics[-1]
            if (
                latest.memory_percent
                > self.monitoring_config.alert_memory_threshold_percent
            ):
                memory_status = "warning"
                status = "warning"
            if latest.memory_mb > self.resource_limits.max_memory_usage_mb:
                memory_status = "critical"
                status = "critical"

        if active_jobs >= self.resource_limits.max_concurrent_jobs:
            processing_status = "critical"
            status = "critical"
        elif active_jobs > self.resource_limits.max_concurrent_jobs * 0.8:
            processing_status = "warning"
            if status == "healthy":
                status = "warning"

        return SystemHealth(
            timestamp=current_time,
            status=status,
            memory_status=memory_status,
            processing_status=processing_status,
            alerts=recent_alerts,
            active_jobs=active_jobs,
            queue_size=0,  # Could be extended to track actual queue
        )

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get a summary of metrics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            recent_resource = [
                m for m in self._metrics_history if m.timestamp > cutoff_time
            ]
            recent_performance = [
                m for m in self._performance_history if m.start_time > cutoff_time
            ]

        summary = {
            "time_period_hours": hours,
            "resource_metrics": {
                "count": len(recent_resource),
                "memory_avg_mb": sum(m.memory_mb for m in recent_resource)
                / max(len(recent_resource), 1),
                "memory_max_mb": max((m.memory_mb for m in recent_resource), default=0),
                "cpu_avg_percent": sum(m.cpu_percent for m in recent_resource)
                / max(len(recent_resource), 1),
                "cpu_max_percent": max(
                    (m.cpu_percent for m in recent_resource), default=0
                ),
            },
            "performance_metrics": {
                "total_operations": len(recent_performance),
                "successful_operations": sum(
                    1 for m in recent_performance if m.success
                ),
                "failed_operations": sum(
                    1 for m in recent_performance if not m.success
                ),
                "avg_duration_seconds": sum(
                    m.duration_seconds or 0 for m in recent_performance
                )
                / max(len(recent_performance), 1),
                "max_duration_seconds": max(
                    (m.duration_seconds or 0 for m in recent_performance), default=0
                ),
                "total_pages_processed": sum(
                    m.pages_processed or 0 for m in recent_performance
                ),
            },
            "system_health": asdict(self.get_system_health()),
        }

        return summary

    async def save_metrics_to_file(self, file_path: Optional[str] = None):
        """Save current metrics to a JSON file."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                self.paths_config.metrics_dir, f"metrics_{timestamp}.json"
            )

        summary = self.get_metrics_summary(hours=24)  # Last 24 hours

        try:
            with open(file_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Metrics saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving metrics to {file_path}: {e}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def initialize_monitoring(config: Any) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(config)
    return _metrics_collector


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the global metrics collector instance."""
    return _metrics_collector


async def shutdown_monitoring():
    """Shutdown the global metrics collector."""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.stop()
        _metrics_collector = None
