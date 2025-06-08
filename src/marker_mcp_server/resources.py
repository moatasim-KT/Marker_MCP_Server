"""Resource management for Marker MCP Server.

This module handles shared resources and configurations used across the MCP server,
including model loading, device management, and common utilities.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from .utils import ResourceError, get_logger

logger = get_logger(__name__)

# Global variables for shared resources
_models: Optional[Dict[str, Any]] = None
_device: Optional[str] = None
_dtype: Optional[torch.dtype] = None
_device_kwargs: Dict[str, Any] = {}
_initialized: bool = False


def initialize_resources(force: bool = False) -> None:
    """Initialize shared resources including device settings and models.

    Args:
        force: If True, force reinitialization even if already initialized.

    Raises:
        ResourceError: If resource initialization fails
    """
    global _device, _dtype, _device_kwargs, _models, _initialized

    if not force and _initialized:
        return

    _initialized = True

    try:
        logger.info("Initializing resources...")

        # Set up device and data type
        _device, _dtype, _device_kwargs = _get_device_settings()
        logger.info(f"Using device: {_device}")
        logger.info(f"Using dtype: {_dtype}")
        logger.debug(f"Device kwargs: {_device_kwargs}")

        # Don't load models during initialization - only when needed
        _models = {}
        logger.info("Device settings initialized (models will be loaded on demand)")

        _initialized = True
        logger.info("Resources initialized successfully")

    except Exception as e:
        error_msg = f"Failed to initialize resources: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _initialized = False
        raise ResourceError(error_msg) from e


def _get_device_settings() -> Tuple[str, Optional[torch.dtype], Dict[str, Any]]:
    """Get the best available device (MPS, CUDA, or CPU) and appropriate settings.

    Returns:
        tuple: (device_name, dtype, device_kwargs)

    Raises:
        ResourceError: If device detection fails
    """
    try:
        logger.debug("Detecting available devices...")

        # Default to CPU
        device = "cpu"
        dtype = None
        device_kwargs: Dict[str, Any] = {}

        # Check for CUDA
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
            device_kwargs = {"device": device, "dtype": dtype}

            # Log CUDA device info
            device_count = torch.cuda.device_count()
            logger.info(f"Found {device_count} CUDA device(s):")
            for i in range(device_count):
                logger.info(f"  {i}: {torch.cuda.get_device_name(i)}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_kwargs = {"device": device}
            logger.info("Using MPS (Apple Silicon) device")
        else:
            logger.warning(
                "No GPU/accelerator found, using CPU. This will be slow for large documents."
            )

        logger.debug(f"Selected device: {device}, dtype: {dtype}")
        return device, dtype, device_kwargs

    except Exception as e:
        error_msg = f"Failed to detect device settings: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ResourceError(error_msg) from e


def _initialize_models() -> Dict[str, Any]:
    """Initialize the models if available.

    Returns:
        Dict[str, Any]: Dictionary of loaded models or empty dict if not available.

    Raises:
        ResourceError: If model loading fails
    """
    try:
        logger.info("Loading models...")

        # Import models here to avoid circular imports
        try:
            from marker.models import create_model_dict as load_all_models
        except ImportError as e:
            error_msg = "Marker models package not available. Make sure marker is installed correctly."
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        # Get device and kwargs
        device = get_device()
        device_kwargs = get_device_kwargs()

        logger.debug(f"Loading models with device={device}, kwargs={device_kwargs}")

        # Load models with GPU optimizations
        try:
            # Apply GPU optimizations to device_kwargs
            from .gpu_optimizer import get_gpu_optimizer

            gpu_optimizer = get_gpu_optimizer()
            optimization_params = gpu_optimizer.optimize_model_loading()

            # Merge optimization parameters with device_kwargs
            optimized_kwargs = {**device_kwargs, **optimization_params}
            logger.info(f"Loading models with optimized parameters: {optimized_kwargs}")

            # Since device_kwargs already contains the device parameter, we don't need to pass it separately
            models = load_all_models(**optimized_kwargs)
            if not models:
                logger.warning("No models were loaded")
                return {}

            logger.info(
                f"Successfully loaded {len(models)} models with GPU optimizations"
            )

            # Log memory usage after model loading
            memory_stats = gpu_optimizer.get_memory_stats()
            logger.info(
                f"GPU memory after model loading: {memory_stats['allocated_memory_mb']:.1f}MB ({memory_stats['utilization_percent']:.1f}%)"
            )

            return models

        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ResourceError(error_msg) from e

    except Exception as e:
        error_msg = f"Unexpected error initializing models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ResourceError(error_msg) from e


def get_models(force_reload: bool = False) -> Dict[str, Any]:
    """Get loaded models, loading them on first access.

    Args:
        force_reload: If True, force reload models even if already loaded.

    Returns:
        Dict[str, Any]: Dictionary of loaded models.

    Note:
        Models are loaded lazily only when first requested to avoid slow startup.
    """
    global _models

    # Ensure resources are initialized first
    if not _initialized:
        try:
            initialize_resources()
        except Exception as e:
            logger.warning(f"Failed to initialize resources: {str(e)}")
            return {}

    # Load models if not already loaded or force reload requested
    if not _models or force_reload:
        try:
            logger.info("Loading models on demand...")
            _models = _initialize_models()
            if _models:
                logger.info(f"Successfully loaded {len(_models)} models")
            else:
                logger.warning("No models were loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            _models = {}

    return _models or {}


def get_device() -> str:
    """Get the current device being used for computation.

    Returns:
        str: The device name (e.g., 'cuda', 'mps', 'cpu')

    Note:
        This will attempt to initialize resources if they haven't been initialized yet.
    """
    global _device

    if not _initialized:
        try:
            initialize_resources()
        except Exception as e:
            logger.warning(
                f"Failed to initialize resources, falling back to CPU: {str(e)}"
            )
            return "cpu"

    return _device or "cpu"


def get_device_kwargs() -> Dict[str, Any]:
    """Get the device-specific keyword arguments.

    Returns:
        dict: Device-specific keyword arguments for model initialization.

    Note:
        This will attempt to initialize resources if they haven't been initialized yet.
    """
    global _device_kwargs

    if not _initialized:
        try:
            initialize_resources()
        except Exception as e:
            logger.warning(
                f"Failed to initialize resources, using default device kwargs: {str(e)}"
            )
            return {}

    return _device_kwargs or {}


def get_dtype() -> Optional[torch.dtype]:
    """Get the current data type being used for computation.

    Returns:
        Optional[torch.dtype]: The data type or None if not applicable.

    Note:
        This will attempt to initialize resources if they haven't been initialized yet.
    """
    global _dtype

    if not _initialized:
        try:
            initialize_resources()
        except Exception as e:
            logger.warning(
                f"Failed to initialize resources, using default dtype: {str(e)}"
            )
            return None

    return _dtype


# Note: Resources are now initialized lazily when first needed
# instead of at module import time to avoid startup delays
