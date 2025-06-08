"""
GPU Memory Management for Marker MCP Server.

This module provides utilities for monitoring and optimizing GPU memory usage
during PDF conversion operations.
"""

import torch
import gc
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    allocated_mb: float
    cached_mb: float
    total_mb: float
    utilization_percent: float
    operation: Optional[str] = None

class GPUMemoryManager:
    """Manages GPU memory usage and optimization during conversion."""
    
    def __init__(self, device_type: str = "auto"):
        self.device_type = self._detect_device_type() if device_type == "auto" else device_type
        self.memory_snapshots: List[MemorySnapshot] = []
        self.peak_memory_mb = 0
        self.memory_threshold_mb = self._get_memory_threshold()
        
        logger.info(f"Initialized GPU memory manager for {self.device_type} device")
        logger.info(f"Memory threshold: {self.memory_threshold_mb:.0f} MB")
    
    def _detect_device_type(self) -> str:
        """Detect the current device type."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _get_memory_threshold(self) -> float:
        """Get memory threshold for cleanup triggers."""
        if self.device_type == "cuda":
            # Use 80% of total GPU memory as threshold
            props = torch.cuda.get_device_properties(0)
            return (props.total_memory / (1024 * 1024)) * 0.8
        elif self.device_type == "mps":
            # Use 60% of estimated GPU memory for MPS
            import psutil
            system_memory = psutil.virtual_memory().total / (1024 * 1024)
            return (system_memory * 0.75) * 0.6  # 60% of 75% of system memory
        return float('inf')  # No limit for CPU
    
    def get_current_memory_usage(self) -> MemorySnapshot:
        """Get current GPU memory usage."""
        allocated_mb = 0
        cached_mb = 0
        total_mb = 0
        
        try:
            if self.device_type == "cuda":
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                cached_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                
            elif self.device_type == "mps":
                allocated_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                cached_mb = 0  # MPS doesn't have cached memory concept
                import psutil
                total_mb = (psutil.virtual_memory().total / (1024 * 1024)) * 0.75
                
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
        
        utilization_percent = (allocated_mb / total_mb * 100) if total_mb > 0 else 0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=allocated_mb,
            cached_mb=cached_mb,
            total_mb=total_mb,
            utilization_percent=utilization_percent
        )
        
        # Update peak memory
        self.peak_memory_mb = max(self.peak_memory_mb, allocated_mb)
        
        return snapshot
    
    def log_memory_usage(self, operation: str = ""):
        """Log current memory usage."""
        snapshot = self.get_current_memory_usage()
        snapshot.operation = operation
        self.memory_snapshots.append(snapshot)
        
        logger.debug(
            f"GPU Memory [{operation}]: {snapshot.allocated_mb:.1f}MB allocated "
            f"({snapshot.utilization_percent:.1f}%), "
            f"{snapshot.cached_mb:.1f}MB cached"
        )
        
        # Keep only last 100 snapshots
        if len(self.memory_snapshots) > 100:
            self.memory_snapshots = self.memory_snapshots[-100:]
    
    def clear_memory_cache(self, force: bool = False):
        """Clear GPU memory cache if needed."""
        if self.device_type == "cpu":
            return
            
        current_usage = self.get_current_memory_usage()
        
        # Clear cache if we're above threshold or forced
        if force or current_usage.allocated_mb > self.memory_threshold_mb:
            try:
                if self.device_type == "cuda":
                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA memory cache")
                elif self.device_type == "mps":
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS memory cache")
                    
                # Force garbage collection
                gc.collect()
                
                # Log memory after cleanup
                after_usage = self.get_current_memory_usage()
                freed_mb = current_usage.allocated_mb - after_usage.allocated_mb
                if freed_mb > 0:
                    logger.info(f"Freed {freed_mb:.1f}MB of GPU memory")
                    
            except Exception as e:
                logger.warning(f"Error clearing memory cache: {e}")
    
    @contextmanager
    def memory_context(self, operation: str):
        """Context manager for tracking memory usage during operations."""
        self.log_memory_usage(f"{operation}_start")
        start_memory = self.get_current_memory_usage().allocated_mb
        
        try:
            yield
        finally:
            end_memory = self.get_current_memory_usage().allocated_mb
            memory_delta = end_memory - start_memory
            
            self.log_memory_usage(f"{operation}_end")
            
            if memory_delta > 0:
                logger.debug(f"Operation '{operation}' used {memory_delta:.1f}MB additional GPU memory")
            
            # Clear cache if memory usage is high
            self.clear_memory_cache()
    
    def optimize_for_batch_processing(self, batch_size: int, operation: str) -> int:
        """Optimize batch size based on current memory usage."""
        if self.device_type == "cpu":
            return batch_size
            
        current_usage = self.get_current_memory_usage()
        available_memory = self.memory_threshold_mb - current_usage.allocated_mb
        
        # Estimate memory per item based on operation
        memory_per_item = {
            'layout': 50,
            'detection': 30,
            'recognition': 25,
            'table': 60,
            'equation': 35
        }
        
        # Find the best match for operation
        mem_per_item = 40  # default
        for op_type, mem in memory_per_item.items():
            if op_type in operation.lower():
                mem_per_item = mem
                break
        
        # Calculate optimal batch size
        max_batch_size = max(1, int(available_memory / mem_per_item))
        optimized_batch_size = min(batch_size, max_batch_size)
        
        if optimized_batch_size < batch_size:
            logger.info(
                f"Reduced batch size from {batch_size} to {optimized_batch_size} "
                f"due to memory constraints (available: {available_memory:.0f}MB)"
            )
        
        return optimized_batch_size
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current = self.get_current_memory_usage()
        
        stats = {
            'current_allocated_mb': current.allocated_mb,
            'current_cached_mb': current.cached_mb,
            'current_utilization_percent': current.utilization_percent,
            'peak_memory_mb': self.peak_memory_mb,
            'total_memory_mb': current.total_mb,
            'available_memory_mb': current.total_mb - current.allocated_mb,
            'memory_threshold_mb': self.memory_threshold_mb,
            'device_type': self.device_type,
            'snapshots_count': len(self.memory_snapshots)
        }
        
        # Add recent memory trend
        if len(self.memory_snapshots) >= 2:
            recent_snapshots = self.memory_snapshots[-10:]  # Last 10 snapshots
            memory_trend = recent_snapshots[-1].allocated_mb - recent_snapshots[0].allocated_mb
            stats['memory_trend_mb'] = memory_trend
        
        return stats
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        self.peak_memory_mb = 0
        logger.debug("Reset peak memory tracking")
    
    def should_trigger_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered."""
        if self.device_type == "cpu":
            return False
            
        current_usage = self.get_current_memory_usage()
        return current_usage.allocated_mb > self.memory_threshold_mb

# Global memory manager instance
_memory_manager: Optional[GPUMemoryManager] = None

def get_memory_manager() -> GPUMemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = GPUMemoryManager()
    return _memory_manager

def log_gpu_memory(operation: str = ""):
    """Convenience function to log GPU memory usage."""
    manager = get_memory_manager()
    manager.log_memory_usage(operation)

def clear_gpu_memory(force: bool = False):
    """Convenience function to clear GPU memory."""
    manager = get_memory_manager()
    manager.clear_memory_cache(force)

@contextmanager
def gpu_memory_context(operation: str):
    """Convenience context manager for GPU memory tracking."""
    manager = get_memory_manager()
    with manager.memory_context(operation):
        yield
