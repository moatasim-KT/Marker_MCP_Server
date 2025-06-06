"""Test configuration and utilities for Marker MCP Server tests."""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

from src.marker_mcp_server.config_schema import AppConfig, ResourceLimits, MonitoringConfig, SecurityConfig


@pytest.fixture
def test_config() -> AppConfig:
    """Create a test configuration with safe defaults."""
    return AppConfig(
        resource_limits=ResourceLimits(
            max_file_size_mb=10,  # Small for tests
            max_memory_usage_mb=100,
            max_processing_time_seconds=30,
            max_concurrent_jobs=2,
            max_queue_size=5
        ),
        monitoring=MonitoringConfig(
            enable_metrics=True,
            metrics_interval_seconds=1,  # Fast for tests
            log_performance=True,
            log_memory_usage=False,  # Reduce noise in tests
            alert_memory_threshold_percent=80.0
        ),
        security=SecurityConfig(
            validate_file_paths=True,
            allowed_file_extensions=['.pdf', '.txt'],  # Add .txt for test files
            allowed_input_dirs=[],  # Will be set in individual tests
            allowed_output_dirs=[]
        )
    )


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create subdirectories
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "cache").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "metrics").mkdir()
        
        yield workspace


@pytest.fixture
def sample_pdf_file(temp_workspace: Path) -> Path:
    """Create a sample PDF file for testing."""
    # Copy the existing sample.pdf if it exists, or create a dummy one
    sample_path = temp_workspace / "input" / "sample.pdf"
    
    existing_sample = Path("tests/sample.pdf")
    if existing_sample.exists():
        shutil.copy2(existing_sample, sample_path)
    else:
        # Create a minimal PDF-like file for testing
        sample_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF")
    
    return sample_path


# Pytest markers for test categories
pytest_slow = pytest.mark.slow
pytest_integration = pytest.mark.integration
pytest_unit = pytest.mark.unit
pytest_security = pytest.mark.security
pytest_performance = pytest.mark.performance


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_test_arguments(operation: str, base_path: Path) -> Dict[str, Any]:
        """Create test arguments for different operations."""
        common_args = {
            "debug": False,
            "output_format": "markdown",
            "use_llm": False
        }
        
        if operation == "single_convert":
            return {
                **common_args,
                "file_path": str(base_path / "input" / "sample.pdf"),
                "output_dir": str(base_path / "output")
            }
        elif operation == "batch_convert":
            return {
                **common_args,
                "in_folder": str(base_path / "input"),
                "output_dir": str(base_path / "output"),
                "max_files": 10
            }
        elif operation == "chunk_convert":
            return {
                **common_args,
                "in_folder": str(base_path / "input"),
                "out_folder": str(base_path / "output"),
                "chunk_size": 5
            }
        elif operation == "batch_pages_convert":
            return {
                **common_args,
                "file_path": str(base_path / "input" / "sample.pdf"),
                "output_dir": str(base_path / "output"),
                "pages_per_chunk": 3
            }
        elif operation == "start_server":
            return {
                "host": "127.0.0.1",
                "port": 8000
            }
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def create_large_file(file_path: Path, size_mb: int):
        """Create a large dummy file for testing size limits."""
        with open(file_path, 'wb') as f:
            # Write dummy data
            chunk_size = 1024 * 1024  # 1MB chunks
            for _ in range(size_mb):
                f.write(b'x' * chunk_size)


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
            self.monitoring = False
            self.monitor_thread = None
            self._peak_memory = 0
            self._start_time = None
        
        def __enter__(self):
            """Context manager entry."""
            self._start_time = time.time()
            self.start_monitoring()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Context manager exit."""
            self.stop_monitoring()
        
        def start_monitoring(self, interval: float = 0.1):
            """Start monitoring system resources."""
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        
        def stop_monitoring(self):
            """Stop monitoring and return collected metrics."""
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            return self.metrics
        
        def get_peak_memory(self):
            """Get peak memory usage in bytes."""
            return self._peak_memory
        
        def get_duration(self):
            """Get monitoring duration in seconds."""
            if self._start_time:
                return time.time() - self._start_time
            return 0
        
        def _monitor_loop(self, interval: float):
            """Background monitoring loop."""
            try:
                import psutil
                process = psutil.Process()
                
                while self.monitoring:
                    try:
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        cpu_percent = process.cpu_percent()
                        
                        # Track peak memory
                        if memory_info.rss > self._peak_memory:
                            self._peak_memory = memory_info.rss
                        
                        self.metrics.append({
                            "timestamp": time.time(),
                            "memory_mb": memory_mb,
                            "cpu_percent": cpu_percent
                        })
                        
                        time.sleep(interval)
                    except Exception:
                        break
            except ImportError:
                # psutil not available, skip monitoring
                pass
    
    return PerformanceMonitor
