import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

class MarkerConfig(BaseModel):
    device: Optional[str] = None
    batch_size: int = Field(default=1)
    max_pages: Optional[int] = None
    parallel_factor: int = Field(default=1)

class ResourceLimits(BaseModel):
    """Resource consumption limits for safety and performance."""
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    max_memory_usage_mb: int = Field(default=4096, description="Maximum memory usage in MB")
    max_processing_time_seconds: int = Field(default=600, description="Maximum processing time per file")
    max_concurrent_jobs: int = Field(default=3, description="Maximum concurrent processing jobs")
    max_queue_size: int = Field(default=10, description="Maximum job queue size")
    
    @field_validator('max_file_size_mb', 'max_memory_usage_mb', 'max_processing_time_seconds')
    def positive_values(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v

class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""
    enable_metrics: bool = Field(default=True, description="Enable performance metrics collection")
    metrics_interval_seconds: int = Field(default=30, description="Metrics collection interval")
    log_performance: bool = Field(default=True, description="Log performance metrics")
    log_memory_usage: bool = Field(default=True, description="Log memory usage")
    log_system_stats: bool = Field(default=False, description="Log system-wide statistics")
    alert_memory_threshold_percent: float = Field(default=85.0, description="Memory usage alert threshold")
    alert_processing_time_threshold_seconds: int = Field(default=300, description="Processing time alert threshold")
    
    @field_validator('metrics_interval_seconds', 'alert_processing_time_threshold_seconds')
    def positive_time_values(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v
    
    @field_validator('alert_memory_threshold_percent')
    def valid_percentage(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('Must be between 0 and 100')
        return v

class SecurityConfig(BaseModel):
    """Security configuration for file access and validation."""
    allowed_input_dirs: List[str] = Field(default_factory=list, description="Allowed input directories")
    allowed_output_dirs: List[str] = Field(default_factory=list, description="Allowed output directories")
    validate_file_paths: bool = Field(default=True, description="Enable file path validation")
    allowed_file_extensions: List[str] = Field(default_factory=lambda: ['.pdf'], description="Allowed file extensions")

class PathsConfig(BaseModel):
    cache_dir: str = Field(default_factory=lambda: os.path.expanduser("~/.cache/marker-mcp"))
    model_dir: str = Field(default_factory=lambda: os.path.expanduser("~/.cache/marker-mcp/models"))
    logs_dir: str = Field(default_factory=lambda: os.path.expanduser("~/.cache/marker-mcp/logs"))
    metrics_dir: str = Field(default_factory=lambda: os.path.expanduser("~/.cache/marker-mcp/metrics"))

class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    marker: MarkerConfig = Field(default_factory=MarkerConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
