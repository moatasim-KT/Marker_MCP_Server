"""
GPU optimization utilities for Marker MCP Server.

This module provides utilities to optimize GPU usage, including dynamic batch sizing,
memory management, and device-specific optimizations.
"""

import torch
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU optimizations."""
    device: str
    device_type: str  # 'cuda', 'mps', 'cpu'
    total_memory_mb: float
    available_memory_mb: float
    memory_fraction: float = 0.8
    enable_dynamic_batching: bool = True
    prefetch_factor: int = 2

class GPUOptimizer:
    """Optimizes GPU usage for different operations and devices."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gpu_config = self._detect_gpu_configuration()
        self._optimization_cache = {}
        
    def _detect_gpu_configuration(self) -> GPUOptimizationConfig:
        """Detect and configure GPU settings."""
        device_type = "cpu"
        device = "cpu"
        total_memory_mb = 0
        available_memory_mb = 0
        
        try:
            if torch.cuda.is_available():
                device_type = "cuda"
                device = "cuda"
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory_mb = gpu_props.total_memory / (1024 * 1024)
                available_memory_mb = (gpu_props.total_memory - torch.cuda.memory_allocated()) / (1024 * 1024)
                logger.info(f"CUDA GPU detected: {gpu_props.name}, {total_memory_mb:.0f}MB total")
                
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = "mps"
                device = "mps"
                # For MPS, estimate based on system memory (Apple Silicon shares memory)
                system_memory = psutil.virtual_memory()
                total_memory_mb = system_memory.total / (1024 * 1024) * 0.75  # 75% of system memory
                current_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                available_memory_mb = total_memory_mb - current_allocated
                logger.info(f"MPS GPU detected: Apple Silicon, estimated {total_memory_mb:.0f}MB available")
                
            else:
                logger.warning("No GPU detected, using CPU")
                
        except Exception as e:
            logger.error(f"Error detecting GPU configuration: {e}")
            
        return GPUOptimizationConfig(
            device=device,
            device_type=device_type,
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            memory_fraction=self.config.get('gpu_memory_fraction', 0.8),
            enable_dynamic_batching=self.config.get('dynamic_batch_sizing', True),
            prefetch_factor=self.config.get('prefetch_factor', 2)
        )
    
    def get_optimal_batch_size(self, operation: str, base_batch_size: int = 1) -> int:
        """Calculate optimal batch size for a given operation."""
        cache_key = f"{operation}_{base_batch_size}"
        if cache_key in self._optimization_cache:
            return self._optimization_cache[cache_key]
            
        optimal_size = self._calculate_batch_size(operation, base_batch_size)
        self._optimization_cache[cache_key] = optimal_size
        
        logger.debug(f"Optimal batch size for {operation}: {optimal_size} (base: {base_batch_size})")
        return optimal_size
    
    def _calculate_batch_size(self, operation: str, base_batch_size: int) -> int:
        """Calculate batch size based on operation type and available GPU memory."""
        if not self.gpu_config.enable_dynamic_batching or self.gpu_config.device_type == "cpu":
            return base_batch_size
            
        # Operation-specific memory requirements (MB per item in batch)
        memory_per_item = {
            'layout_detection': 50,      # Layout model memory usage
            'text_detection': 30,        # Text detection memory usage  
            'text_recognition': 25,      # OCR memory usage
            'table_detection': 60,       # Table detection memory usage
            'table_recognition': 40,     # Table OCR memory usage
            'equation_recognition': 35,  # Equation recognition memory usage
            'default': 40               # Default memory estimate
        }
        
        mem_per_item = memory_per_item.get(operation, memory_per_item['default'])
        available_memory = self.gpu_config.available_memory_mb * self.gpu_config.memory_fraction
        
        # Calculate maximum batch size based on memory
        max_batch_size = max(1, int(available_memory / mem_per_item))
        
        # Apply device-specific multipliers
        if self.gpu_config.device_type == "cuda":
            # CUDA can handle larger batches efficiently
            multiplier = 2.0
        elif self.gpu_config.device_type == "mps":
            # MPS is more memory-constrained but can still benefit from larger batches
            multiplier = 1.5
        else:
            multiplier = 1.0
            
        optimal_batch_size = min(max_batch_size, int(base_batch_size * multiplier))
        
        # Ensure minimum batch size of 1
        return max(1, optimal_batch_size)
    
    def get_device_specific_batch_sizes(self) -> Dict[str, int]:
        """Get optimized batch sizes for all operations."""
        operations = [
            'layout_detection',
            'text_detection', 
            'text_recognition',
            'table_detection',
            'table_recognition',
            'equation_recognition'
        ]
        
        # Base batch sizes (conservative defaults)
        base_sizes = {
            'layout_detection': 6,
            'text_detection': 4,
            'text_recognition': 32,
            'table_detection': 4,
            'table_recognition': 6,
            'equation_recognition': 6
        }
        
        optimized_sizes = {}
        for operation in operations:
            base_size = base_sizes.get(operation, 4)
            optimized_sizes[operation] = self.get_optimal_batch_size(operation, base_size)
            
        return optimized_sizes
    
    def optimize_model_loading(self) -> Dict[str, Any]:
        """Get optimized model loading parameters."""
        optimization_params = {
            'device': self.gpu_config.device,
            'dtype': torch.float32,  # Default
        }
        
        if self.gpu_config.device_type == "cuda":
            # Use bfloat16 for CUDA if supported
            if torch.cuda.is_bf16_supported():
                optimization_params['dtype'] = torch.bfloat16
            else:
                optimization_params['dtype'] = torch.float16
                
        elif self.gpu_config.device_type == "mps":
            # MPS works best with float32
            optimization_params['dtype'] = torch.float32
            
        # Add memory management settings
        if self.gpu_config.device_type in ["cuda", "mps"]:
            optimization_params.update({
                'memory_fraction': self.gpu_config.memory_fraction,
                'enable_memory_efficient_attention': True,
                'gradient_checkpointing': False,  # Not needed for inference
            })
            
        return optimization_params
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        stats = {
            'total_memory_mb': self.gpu_config.total_memory_mb,
            'available_memory_mb': self.gpu_config.available_memory_mb,
            'allocated_memory_mb': 0,
            'cached_memory_mb': 0,
            'utilization_percent': 0
        }
        
        try:
            if self.gpu_config.device_type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                cached = torch.cuda.memory_reserved() / (1024 * 1024)
                stats.update({
                    'allocated_memory_mb': allocated,
                    'cached_memory_mb': cached,
                    'utilization_percent': (allocated / self.gpu_config.total_memory_mb) * 100
                })
                
            elif self.gpu_config.device_type == "mps":
                allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                stats.update({
                    'allocated_memory_mb': allocated,
                    'utilization_percent': (allocated / self.gpu_config.total_memory_mb) * 100
                })
                
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")
            
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        try:
            if self.gpu_config.device_type == "cuda":
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            elif self.gpu_config.device_type == "mps":
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
        except Exception as e:
            logger.warning(f"Error clearing GPU cache: {e}")

# Global optimizer instance
_gpu_optimizer: Optional[GPUOptimizer] = None

def get_gpu_optimizer(config: Optional[Dict[str, Any]] = None) -> GPUOptimizer:
    """Get the global GPU optimizer instance."""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer(config)
    return _gpu_optimizer

def optimize_batch_sizes_for_device() -> Dict[str, int]:
    """Get optimized batch sizes for the current device."""
    optimizer = get_gpu_optimizer()
    return optimizer.get_device_specific_batch_sizes()
