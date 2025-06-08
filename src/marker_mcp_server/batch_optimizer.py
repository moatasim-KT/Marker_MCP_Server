"""
Batch size optimization patches for Marker processors.

This module provides optimized batch sizes for different Marker processors
based on the available GPU and memory configuration.
"""

import logging
from typing import Dict, Any, Optional
from .gpu_optimizer import get_gpu_optimizer

logger = logging.getLogger(__name__)

class BatchSizeOptimizer:
    """Optimizes batch sizes for Marker processors based on GPU capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.gpu_optimizer = get_gpu_optimizer(config)
        self.optimized_sizes = self.gpu_optimizer.get_device_specific_batch_sizes()
        self.device_type = self.gpu_optimizer.gpu_config.device_type
        
        logger.info(f"Initialized batch optimizer for {self.device_type} device")
        logger.info(f"Optimized batch sizes: {self.optimized_sizes}")
    
    def get_layout_batch_size(self, default: int = 6) -> int:
        """Get optimized batch size for layout detection."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('layout_detection', 12))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('layout_detection', 8))
        return default
    
    def get_detection_batch_size(self, default: int = 4) -> int:
        """Get optimized batch size for text detection."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('text_detection', 12))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('text_detection', 6))
        return default
    
    def get_recognition_batch_size(self, default: int = 32) -> int:
        """Get optimized batch size for text recognition."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('text_recognition', 64))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('text_recognition', 48))
        return default
    
    def get_table_detection_batch_size(self, default: int = 4) -> int:
        """Get optimized batch size for table detection."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('table_detection', 12))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('table_detection', 6))
        return default
    
    def get_table_rec_batch_size(self, default: int = 6) -> int:
        """Get optimized batch size for table recognition."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('table_recognition', 14))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('table_recognition', 10))
        return default
    
    def get_equation_batch_size(self, default: int = 6) -> int:
        """Get optimized batch size for equation recognition."""
        if self.device_type == "cuda":
            return max(default, self.optimized_sizes.get('equation_recognition', 16))
        elif self.device_type == "mps":
            return max(default, self.optimized_sizes.get('equation_recognition', 10))
        return default
    
    def get_ocr_error_batch_size(self, default: int = 4) -> int:
        """Get optimized batch size for OCR error detection."""
        if self.device_type == "cuda":
            return max(default, 14)
        elif self.device_type == "mps":
            return max(default, 8)
        return default

# Global batch optimizer instance
_batch_optimizer: Optional[BatchSizeOptimizer] = None

def get_batch_optimizer(config: Optional[Dict[str, Any]] = None) -> BatchSizeOptimizer:
    """Get the global batch optimizer instance."""
    global _batch_optimizer
    if _batch_optimizer is None:
        _batch_optimizer = BatchSizeOptimizer(config)
    return _batch_optimizer

def patch_processor_batch_sizes():
    """Patch Marker processor classes to use optimized batch sizes."""
    try:
        optimizer = get_batch_optimizer()
        
        # Import marker modules
        from marker.processors.equation import EquationProcessor
        from marker.processors.table import TableProcessor
        from marker.builders.line import LineBuilder
        from marker.builders.layout import LayoutBuilder
        
        # Patch EquationProcessor
        original_eq_get_batch_size = EquationProcessor.get_batch_size
        def optimized_eq_get_batch_size(self):
            default = original_eq_get_batch_size(self)
            return optimizer.get_equation_batch_size(default)
        EquationProcessor.get_batch_size = optimized_eq_get_batch_size
        
        # Patch TableProcessor
        original_table_detection_batch_size = TableProcessor.get_detection_batch_size
        def optimized_table_detection_batch_size(self):
            default = original_table_detection_batch_size(self)
            return optimizer.get_table_detection_batch_size(default)
        TableProcessor.get_detection_batch_size = optimized_table_detection_batch_size
        
        original_table_rec_batch_size = TableProcessor.get_table_rec_batch_size
        def optimized_table_rec_batch_size(self):
            default = original_table_rec_batch_size(self)
            return optimizer.get_table_rec_batch_size(default)
        TableProcessor.get_table_rec_batch_size = optimized_table_rec_batch_size
        
        original_table_recognition_batch_size = TableProcessor.get_recognition_batch_size
        def optimized_table_recognition_batch_size(self):
            default = original_table_recognition_batch_size(self)
            return optimizer.get_recognition_batch_size(default)
        TableProcessor.get_recognition_batch_size = optimized_table_recognition_batch_size
        
        # Patch LineBuilder
        original_line_detection_batch_size = LineBuilder.get_detection_batch_size
        def optimized_line_detection_batch_size(self):
            default = original_line_detection_batch_size(self)
            return optimizer.get_detection_batch_size(default)
        LineBuilder.get_detection_batch_size = optimized_line_detection_batch_size
        
        original_line_ocr_error_batch_size = LineBuilder.get_ocr_error_batch_size
        def optimized_line_ocr_error_batch_size(self):
            default = original_line_ocr_error_batch_size(self)
            return optimizer.get_ocr_error_batch_size(default)
        LineBuilder.get_ocr_error_batch_size = optimized_line_ocr_error_batch_size
        
        # Patch LayoutBuilder
        original_layout_get_batch_size = LayoutBuilder.get_batch_size
        def optimized_layout_get_batch_size(self):
            default = original_layout_get_batch_size(self)
            return optimizer.get_layout_batch_size(default)
        LayoutBuilder.get_batch_size = optimized_layout_get_batch_size
        
        logger.info("Successfully patched processor batch sizes for GPU optimization")
        
        # Log the optimized batch sizes
        device_type = optimizer.device_type
        logger.info(f"Optimized batch sizes for {device_type}:")
        logger.info(f"  Layout detection: {optimizer.get_layout_batch_size()}")
        logger.info(f"  Text detection: {optimizer.get_detection_batch_size()}")
        logger.info(f"  Text recognition: {optimizer.get_recognition_batch_size()}")
        logger.info(f"  Table detection: {optimizer.get_table_detection_batch_size()}")
        logger.info(f"  Table recognition: {optimizer.get_table_rec_batch_size()}")
        logger.info(f"  Equation recognition: {optimizer.get_equation_batch_size()}")
        
    except ImportError as e:
        logger.warning(f"Could not patch processor batch sizes - marker modules not available: {e}")
    except Exception as e:
        logger.error(f"Error patching processor batch sizes: {e}")

def apply_gpu_optimizations(config: Optional[Dict[str, Any]] = None):
    """Apply all GPU optimizations."""
    logger.info("Applying GPU optimizations...")
    
    # Initialize optimizer
    optimizer = get_gpu_optimizer(config)
    
    # Log GPU configuration
    gpu_config = optimizer.gpu_config
    logger.info(f"GPU Configuration:")
    logger.info(f"  Device: {gpu_config.device} ({gpu_config.device_type})")
    logger.info(f"  Total Memory: {gpu_config.total_memory_mb:.0f} MB")
    logger.info(f"  Available Memory: {gpu_config.available_memory_mb:.0f} MB")
    logger.info(f"  Memory Fraction: {gpu_config.memory_fraction}")
    
    # Get memory stats
    memory_stats = optimizer.get_memory_stats()
    logger.info(f"  Current Utilization: {memory_stats['utilization_percent']:.1f}%")
    
    # Patch batch sizes
    patch_processor_batch_sizes()
    
    # Clear GPU cache to start fresh
    optimizer.clear_gpu_cache()
    
    logger.info("GPU optimizations applied successfully")
