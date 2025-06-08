# Marker MCP Server - Enhanced PDF Processing

An advanced MCP (Model Context Protocol) server for high-quality PDF to Markdown conversion with comprehensive monitoring, security, and testing capabilities.

## 🎯 Project Overview

This implementation provides a comprehensive MCP server with advanced features including:
- Real-time monitoring and metrics collection
- Advanced security framework
- Comprehensive testing suite
- High-performance PDF processing
- LLM integration for enhanced output quality
- Batch and chunked processing capabilities

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install .

# Or using poetry
poetry install
```

### Basic Usage

```bash
# Start the MCP server
python -m marker_mcp_server

# Show help and available options
python -m src.marker_mcp_server.server --help

# Show version information
python -m src.marker_mcp_server.server --version

# Enable debug logging
python -m src.marker_mcp_server.server --debug
```

## 🛠️ MCP Tools Available

### 1. `batch_pages_convert` - Advanced Chunked Processing

**NEW FEATURE**: Process large PDFs efficiently by splitting them into page chunks.

- **Memory Efficient**: Processes documents in configurable page chunks (default: 5 pages)
- **Fault Tolerant**: Individual chunk failures don't stop entire process
- **Progress Tracking**: Detailed progress information for each chunk
- **Automatic Stitching**: Combines chunk outputs into single cohesive document

```python
# Example usage
arguments = {
    "file_path": "/path/to/large_document.pdf",
    "pages_per_chunk": 5,
    "combine_output": True,
    "use_llm": True,
    "output_format": "markdown"
}
```

### 2. `batch_convert` - Enhanced Batch Processing

Convert multiple PDFs in a folder with full CLI argument support.

```python
arguments = {
    "folder_path": "/path/to/pdfs",
    "output_dir": "/path/to/outputs",
    "workers": 8,
    "debug": True,
    "use_llm": True,
    "page_range": "0-10",
    "skip_existing": True
}
```

### 3. `single_convert` - Single File Conversion

Convert individual PDF files with advanced options.

```python
arguments = {
    "pdf_path": "/path/to/document.pdf",
    "output_path": "/path/to/output.md",
    "debug": True,
    "use_llm": True,
    "page_range": "0-5"
}
```

### 4. `chunk_convert` - Folder Chunking

Process large collections of PDFs using memory-efficient chunking.

```python
arguments = {
    "in_folder": "/path/to/large/collection",
    "chunk_size": 50,
    "use_llm": True
}
```

### 5. `start_server` - API Server

Start FastAPI server for REST API access.

```python
arguments = {
    "host": "0.0.0.0",
    "port": 8080
}
```

## 🔧 Advanced Configuration

### LLM Integration

Enable high-quality processing with Large Language Models:

```python
# Basic LLM usage
{
    "use_llm": True,
    "llm_service": "groq"  # Automatically normalized to full path
}

# Advanced LLM configuration
{
    "use_llm": True,
    "llm_service": "marker.services.groq.GroqService",
    "config_json": "examples/llm_enhanced_config.json"
}
```

### Page Range Selection

Process specific page ranges efficiently:

```python
{
    "page_range": "0-5",      # Pages 0 through 5
    "page_range": "0,3,5-10", # Pages 0, 3, and 5 through 10
    "page_range": "10-"       # Page 10 to end
}
```

### Output Formats

Choose from multiple output formats:

```python
{
    "output_format": "markdown",  # Default, clean markdown
    "output_format": "json",      # Structured JSON with metadata
    "output_format": "html"       # Styled HTML output
}
```

### Debug Mode

Enable comprehensive debugging:

```python
{
    "debug": True  # Saves debug images, processing data, and detailed logs
}
```

## 📁 Configuration Files

Use JSON configuration files for complex setups:

### Basic Configuration
```json
{
  "use_llm": false,
  "output_format": "markdown",
  "debug": false,
  "extract_images": true,
  "pdftext_workers": 2
}
```

### LLM-Enhanced Configuration
```json
{
  "use_llm": true,
  "llm_service": "marker.services.groq.GroqService",
  "output_format": "markdown",
  "debug": false,
  "extract_images": true,
  "format_lines": true
}
```

### High-Performance Configuration
```json
{
  "workers": 8,
  "max_tasks_per_worker": 20,
  "disable_multiprocessing": false,
  "pdftext_workers": 4,
  "chunk_size": 100
}
```

## 📊 Performance Metrics

### System Health Monitoring

- **Memory Usage**: Real-time tracking with configurable alerts (85% threshold)
- **CPU Utilization**: Multi-core usage monitoring
- **GPU Usage**: Apple Silicon MPS device monitoring
- **Processing Times**: Per-operation timing with alert thresholds (300s)

### Operation Tracking

- **Job Lifecycle**: Start, progress, completion, and error states
- **Resource Consumption**: Memory, CPU, and GPU usage per operation
- **Throughput**: Pages per second and batch processing metrics
- **Error Rates**: Failure tracking and categorization

## 🛡️ Security Features

### File System Protection

- **Path Traversal Prevention**: Blocks `../` and absolute path attacks
- **Directory Restriction**: Enforces allowed input/output directories
- **Extension Validation**: Restricts to approved file types (`.pdf`)
- **Filename Sanitization**: Prevents malicious filename patterns

### Input Validation

- **Parameter Sanitization**: Type checking and range validation
- **Configuration Validation**: Schema-based security settings
- **Access Logging**: Detailed security event tracking

## 🧪 Testing Coverage

### Test Infrastructure

- **Fixtures**: Configuration, temporary workspace, mock collectors
- **Test Data Generation**: Synthetic performance data and test scenarios
- **Environment Setup**: Isolated test environments with cleanup

### Test Categories

- **Unit Tests**: Component-level testing with mocking
- **Integration Tests**: End-to-end workflow validation
- **Security Tests**: Attack scenario prevention
- **Performance Tests**: Load testing and benchmarking capabilities

## 🚀 Usage Examples

### Basic MCP Client Usage

```python
# Convert PDF with monitoring
result = await mcp_client.call_tool("convert_single_pdf", {
    "file_path": "/safe/path/document.pdf",
    "output_format": "markdown"
})

# Check system health
health = await mcp_client.call_tool("get_system_health", {})

# Get performance metrics
metrics = await mcp_client.call_tool("get_metrics_summary", {})
```

### Configuration

```json
{
    "resource_limits": {
        "max_file_size_mb": 500,
        "max_memory_usage_mb": 4096,
        "max_processing_time_seconds": 600,
        "max_concurrent_jobs": 3
    },
    "monitoring": {
        "enable_metrics": true,
        "metrics_interval_seconds": 30,
        "alert_memory_threshold_percent": 85.0
    },
    "security": {
        "validate_file_paths": true,
        "allowed_input_dirs": ["/safe/input"],
        "allowed_output_dirs": ["/safe/output"]
    }
}
```

## 📈 Performance Characteristics

### Throughput
- **Single PDF**: ~2-5 pages/second (device dependent)
- **Batch Processing**: 3 concurrent jobs by default
- **Memory Efficient**: Streaming processing for large files

### Resource Usage
- **Memory**: Configurable limits with real-time monitoring
- **CPU**: Multi-core utilization with Apple Silicon optimization
- **GPU**: MPS acceleration on compatible devices
- **Storage**: Efficient caching and cleanup

## 🔄 Development Workflow

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -m "unit"
python -m pytest tests/ -m "security"
python -m pytest tests/ -m "performance"

# Run with coverage
python -m pytest tests/ --cov=src/marker_mcp_server
```

### Development Server

```bash
# Start development server
python -m src.marker_mcp_server.server --debug

# With custom configuration
python -m src.marker_mcp_server.server --config-path /path/to/config.json
```

## 📖 Documentation

### Detailed Documentation

For more detailed documentation on:
- Configuration options
- Advanced usage
- API endpoints
- Development guidelines

Please refer to the project's documentation in the `docs/` directory.

## 🤝 Contributing

1. **Fork the repository** and create your branch from `main`.
2. **Write tests** for your changes (see `tests/` directory).
3. **Document** new features or changes in the README or relevant doc files.
4. **Open a Pull Request** with a clear description of your changes.

## 📄 License

This project is licensed under the terms of the [Marker Contributor Agreement](CLA.md).

## 🙏 Acknowledgments

Special thanks to all contributors who have helped make this project possible.

## 📬 Support

For support, please open an issue in the GitHub repository or contact the maintainers directly.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install .

# Or using poetry
poetry install
```

### Basic Usage

```bash
# Start the MCP server
python -m marker_mcp_server

# Show help and available options
python -m src.marker_mcp_server.server --help

# Show version information
python -m src.marker_mcp_server.server --version

# Enable debug logging
python -m src.marker_mcp_server.server --debug
```

## 🛠️ MCP Tools Available

### 1. `batch_pages_convert` - 🆕 Advanced Chunked Processing

**NEW FEATURE**: Process large PDFs efficiently by splitting them into page chunks.

- **Memory Efficient**: Processes documents in configurable page chunks (default: 5 pages)
- **Fault Tolerant**: Individual chunk failures don't stop entire process
- **Progress Tracking**: Detailed progress information for each chunk
- **Automatic Stitching**: Combines chunk outputs into single cohesive document

```python
# Example usage
arguments = {
    "file_path": "/path/to/large_document.pdf",
    "pages_per_chunk": 5,
    "combine_output": True,
    "use_llm": True,
    "output_format": "markdown"
}
```

### 2. `batch_convert` - Enhanced Batch Processing

Convert multiple PDFs in a folder with full CLI argument support.

```python
arguments = {
    "folder_path": "/path/to/pdfs",
    "output_dir": "/path/to/outputs",
    "workers": 8,
    "debug": True,
    "use_llm": True,
    "page_range": "0-10",
    "skip_existing": True
}
```

### 3. `single_convert` - Single File Conversion

Convert individual PDF files with advanced options.

```python
arguments = {
    "pdf_path": "/path/to/document.pdf",
    "output_path": "/path/to/output.md",
    "debug": True,
    "use_llm": True,
    "page_range": "0-5"
}
```

### 4. `chunk_convert` - Folder Chunking

Process large collections of PDFs using memory-efficient chunking.

```python
arguments = {
    "in_folder": "/path/to/large/collection",
    "chunk_size": 50,
    "use_llm": True
}
```

### 5. `start_server` - API Server

Start FastAPI server for REST API access.

```python
arguments = {
    "host": "0.0.0.0",
    "port": 8080
}
```

## 🔧 Advanced Configuration

### LLM Integration

Enable high-quality processing with Large Language Models:

```python
# Basic LLM usage
{
    "use_llm": True,
    "llm_service": "groq"  # Automatically normalized to full path
}

# Advanced LLM configuration
{
    "use_llm": True,
    "llm_service": "marker.services.groq.GroqService",
    "config_json": "examples/llm_enhanced_config.json"
}
```

### Page Range Selection

Process specific page ranges efficiently:

```python
{
    "page_range": "0-5",      # Pages 0 through 5
    "page_range": "0,3,5-10", # Pages 0, 3, and 5 through 10
    "page_range": "10-"       # Page 10 to end
}
```

### Output Formats

Choose from multiple output formats:

```python
{
    "output_format": "markdown",  # Default, clean markdown
    "output_format": "json",      # Structured JSON with metadata
    "output_format": "html"       # Styled HTML output
}
```

### Debug Mode

Enable comprehensive debugging:

```python
{
    "debug": True  # Saves debug images, processing data, and detailed logs
}
```

## 📁 Configuration Files

Use JSON configuration files for complex setups:

### Basic Configuration
```json
{
  "use_llm": false,
  "output_format": "markdown",
  "debug": false,
  "extract_images": true,
  "pdftext_workers": 2
}
```

### LLM-Enhanced Configuration
```json
{
  "use_llm": true,
  "llm_service": "marker.services.groq.GroqService",
  "output_format": "markdown",
  "debug": false,
  "extract_images": true,
  "format_lines": true
}
```

### High-Performance Configuration
```json
{
  "workers": 8,
  "max_tasks_per_worker": 20,
  "disable_multiprocessing": false,
  "pdftext_workers": 4,
  "chunk_size": 100
}
```

## 📚 API Documentation

- When the server is started with `start_server`, FastAPI automatically exposes OpenAPI/Swagger documentation at `/docs` (Swagger UI) and `/redoc` (ReDoc UI).
- You can interactively test the API and see all endpoints and schemas there.

## 🧩 Extending Processors and Converters

- To add a new processor, use the plugin registry in `marker/processors/registry.py`:

```python
from marker.processors import register_processor

@register_processor('my_custom_processor')
class MyCustomProcessor:
    ...
```

- To add a new converter, use the plugin registry in `marker/converters/registry.py`:

```python
from marker.converters import register_converter

@register_converter('my_custom_converter')
class MyCustomConverter:
    ...
```

- See `marker/processors/__init__.py` and `marker/converters/__init__.py` for more details.

## ⚡ Performance & Scalability

- Batch and chunked processing are supported for large-scale PDF conversion.
- For very large jobs or distributed processing, consider integrating an async task queue (e.g., Celery, RQ). This is not included by default, but the architecture supports async handlers.
- Monitor resource usage (CPU, memory) for large jobs. Logging includes memory usage if `psutil` is installed.
- You can adjust worker counts and chunk sizes in the configuration for optimal performance.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing, adding new processors/converters, and running tests.
