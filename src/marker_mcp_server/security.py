"""Security validation and file path protection for Marker MCP Server.

This module provides security features including:
- File path validation and traversal protection
- Input sanitization
- Resource access controls
- Security logging
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Any
import re
import unicodedata

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class SecurityValidator:
    """Handles security validation for file operations and inputs."""
    
    def __init__(self, config: Any):
        self.config = config
        self.security_config = config.security
        
    def validate_file_path(self, file_path: Union[str, Path],
                          operation: str = "read",
                          is_directory: bool = False,
                          skip_extension_check: bool = False) -> Path:
        """Validate a file path for security and access permissions.

        Args:
            file_path: The file path to validate
            operation: Type of operation ('read', 'write')
            is_directory: Whether the path is expected to be a directory
            skip_extension_check: Whether to skip file extension validation

        Returns:
            Validated and resolved Path object

        Raises:
            SecurityError: If the path is invalid or not allowed
        """
        if not self.security_config.validate_file_paths:
            return Path(file_path).resolve()

        try:
            # Check directory traversal protection BEFORE resolving
            self._check_directory_traversal_raw(str(file_path))

            # Convert to Path and resolve
            path = Path(file_path).resolve()

            # Check if file/directory exists for read operations
            if operation == "read" and not is_directory and not path.exists():
                raise SecurityError(f"File does not exist: {path}")
            if operation == "read" and is_directory and not path.exists():
                # For directories in read mode, it's okay if it doesn't exist yet if we are not writing
                pass # Or raise SecurityError(f"Directory does not exist: {path}") if strict
            
            # Check file extension (skip for directories and when explicitly disabled)
            if not is_directory and not skip_extension_check:
                if path.suffix.lower() not in self.security_config.allowed_file_extensions:
                    allowed = ", ".join(self.security_config.allowed_file_extensions)
                    raise SecurityError(f"File extension {path.suffix} not allowed. Allowed: {allowed}")

            # Check directory traversal protection on resolved path
            self._check_directory_traversal(path, operation)

            # Check against allowed directories
            if operation == "read" and self.security_config.allowed_input_dirs:
                self._check_allowed_directories(path, self.security_config.allowed_input_dirs, "input")
            elif operation == "write" and self.security_config.allowed_output_dirs:
                # For write operations, the path or its parent should be in allowed_output_dirs
                self._check_allowed_directories(path, self.security_config.allowed_output_dirs, "output", is_target_for_write=True)

            logger.debug(f"Path validation successful: {path} ({operation})")
            return path

        except Exception as e:
            logger.warning(f"Path validation failed for {file_path}: {e}")
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Invalid file path: {e}")

    def _check_directory_traversal_raw(self, raw_path_str: str):
        """Check for directory traversal attempts in raw path string."""
        # Check for suspicious patterns before path resolution
        # More aggressive patterns for raw string
        suspicious_patterns = [
            r'\\.\\.', # Windows device paths / UNC paths like \\.\ or \\?\
            r'\.\.[/\\]',  # Parent directory references
            r'[/\\]\.\.[/\\]', # Traversal in middle
            r'[/\\]\.\.$', # Traversal at end
            r'^\.\.[/\\]', # Traversal at start
            r'^\.\.$', # Just ".."
            r'%2e%2e%2f', # URL encoded ../
            r'%2e%2e/', # URL encoded ../
            r'..%2f', # URL encoded ../
            r'%2e%2e%5c', # URL encoded ..\\
            r'..%5c', # URL encoded ..\\
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, raw_path_str, re.IGNORECASE):
                logger.error(f"Potential directory traversal in raw path: {raw_path_str} (pattern: {pattern})")
                raise SecurityError(f"Directory traversal detected in path: {raw_path_str}")

    def _check_directory_traversal(self, path: Path, operation: str):
        """Check for directory traversal attempts."""
        path_str = str(path)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.\.[\\/]',  # Parent directory references
            r'[\\/]\.\.[\\/]',  # Directory traversal in middle
            r'[\\/]\.\.$',  # Directory traversal at end
            r'^\.\.[\\/]',  # Directory traversal at start
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, path_str):
                raise SecurityError(f"Directory traversal detected in path: {path}")
        
        # Additional check: ensure resolved path doesn't contain ".." components
        if ".." in path.parts:
            raise SecurityError(f"Directory traversal detected in resolved path: {path}")
    
    def _check_allowed_directories(self, path: Path, allowed_dirs: List[str], dir_type: str, is_target_for_write: bool = False):
        """Check if path is within allowed directories."""
        if not allowed_dirs:
            return  # No restrictions if list is empty

        resolved_allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        
        path_to_check = path.parent if is_target_for_write and not path.is_dir() else path

        is_allowed = any(
            path_to_check == allowed_dir or path_to_check.is_relative_to(allowed_dir)
            for allowed_dir in resolved_allowed_dirs
        )

        if not is_allowed:
            raise SecurityError(
                f"Path {path} is not within allowed {dir_type} directories: {allowed_dirs}"
            )

    def validate_output_directory(self, output_dir: Union[str, Path]) -> Path:
        """Validate and create an output directory.

        Args:
            output_dir: The output directory path

        Returns:
            Validated and resolved Path object for the directory

        Raises:
            SecurityError: If validation fails or directory cannot be created
        """
        try:
            # Validate as a directory, skip extension check
            output_path = self.validate_file_path(output_dir, operation="write", is_directory=True)

            if output_path.exists() and not output_path.is_dir():
                raise SecurityError(f"Output path exists but is not a directory: {output_path}")
            
            # Create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Validated and ensured output directory exists: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Output directory validation failed: {e}")
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Invalid output directory: {e}")

    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """Sanitize a filename to prevent security issues and ensure compatibility.

        Args:
            filename: The original filename
            max_length: Maximum allowed length for the filename (excluding extension)

        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"

        # Normalize Unicode characters (NFKC is a good general choice)
        filename = unicodedata.normalize('NFKC', filename)

        # Split filename and extension
        base, ext = os.path.splitext(filename)

        # Strip leading/trailing whitespace
        base = base.strip()
        # Remove leading dots (but not trailing dot)
        base = base.lstrip('.')
        # Replace all non-allowed characters with underscores (do not collapse)
        base = re.sub(r'[^A-Za-z0-9_.-]', '_', base)
        # Remove leading/trailing underscores and dashes, but preserve trailing dot if present
        if base.endswith('.') and not base.endswith('..'):
            base = base.rstrip('_-')
        else:
            base = base.strip('_-')
        # If base became empty after truncation and stripping
        if not base:
            base = "truncated_file"
        
        # If base is empty after all sanitization (e.g. input was "...", "___", or became empty after truncation)
        if not base:
            base = "sanitized_file"
            
        # Truncate base if necessary to fit max_length (base + ext <= max_length)
        max_base_length = max_length - len(ext)
        if len(base) > max_base_length:
            base = base[:max_base_length]

        # Reconstruct filename
        sanitized = base + ext

        # Final check for empty or dot-only filenames (e.g. if ext was also empty)
        if not sanitized.strip('.'):
            return "unnamed_file"

        # Strip trailing whitespace after all other cleaning (per test expectation)
        sanitized = sanitized.rstrip()

        return sanitized

    def validate_page_range(self, page_range: Optional[str]) -> Optional[str]:
        """Validate page range format.
        
        Args:
            page_range: Page range string (e.g., "1-5,10,15-20")
            
        Returns:
            Validated page range string or None
            
        Raises:
            SecurityError: If page range format is invalid
        """
        if not page_range:
            return None
        
        # Pattern for valid page range: numbers, commas, dashes, spaces
        if not re.match(r'^[\d\s,\-]+$', page_range):
            raise SecurityError(f"Invalid page range format: {page_range}")
        
        # Additional validation could be added here
        # (e.g., checking for reasonable page numbers)
        
        return page_range.strip()
    
    def validate_config_json_path(self, config_path: Union[str, Path]) -> Path:
        """Validate the path to a configuration JSON file.

        Args:
            config_path: Path to the config.json file

        Returns:
            Validated and resolved Path object

        Raises:
            SecurityError: If validation fails
        """
        if config_path is None:
            return None

        try:
            # Validate as a file, skip extension check initially, then check for .json
            path = self.validate_file_path(config_path, operation="read", skip_extension_check=True)
            if path.suffix.lower() != ".json":
                raise SecurityError(f"Configuration file must be a .json file, got: {path.suffix}")
            return path
        except Exception as e:
            logger.error(f"Config JSON path validation failed: {e}")
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Invalid config JSON path: {e}")
    
    def log_security_event(self, event_type: str, details: str, severity: str = "WARNING"):
        """Log a security-related event.
        
        Args:
            event_type: Type of security event
            details: Details about the event
            severity: Log severity level
        """
        log_msg = f"SECURITY_EVENT: {event_type} - {details}"
        
        if severity.upper() == "CRITICAL":
            logger.critical(log_msg)
        elif severity.upper() == "ERROR":
            logger.error(log_msg)
        elif severity.upper() == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)


def create_security_validator(config: Any) -> SecurityValidator:
    """Create a SecurityValidator instance with the given configuration."""
    return SecurityValidator(config)
