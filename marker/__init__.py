"""Marker PDF conversion library.

This package provides tools for converting PDF documents to various formats
including Markdown, HTML, and JSON with advanced layout detection and text extraction.
"""

__version__ = "0.2.17"

# Import key components to make them available at the package level
from .models import create_model_dict
from .output import save_output
from .settings import settings

# Define public API
__all__ = [
    'create_model_dict',
    'save_output', 
    'settings',
    '__version__',
]
