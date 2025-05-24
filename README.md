# Marker Project: Unified Document Conversion & MCP Server

## Overview
Marker is a unified document conversion and monitoring platform with a modular architecture. It provides:
- **File conversion** (PDF to Markdown and more)
- **Monitoring and performance tracking**
- **MCP (Model Context Protocol) server** for integration with clients like Claude Desktop
- **Configurable tools and resources** for flexible workflows

## Key Components
- **marker_launcher.py**: Unified entry point for all Marker applications (conversion, monitoring, server, etc.)
- **src/mcp_server.py**: Main MCP server logic, exposing resources and tools for file conversion, configuration, and monitoring
- **unified_conversion.py**: Core conversion logic (PDF to Markdown, etc.)
- **unified_monitoring.py**: Monitoring and performance tracking
- **config_fixes/optimized_config.py**: Configuration management (load/save JSON config)
- **conversion_results/**: Output directory for converted files
- **uploads/**: Directory for user-uploaded files

## How the Project Functions
### 1. Launching Applications
Use `marker_launcher.py` to launch any Marker app:
```sh
python marker_launcher.py --list           # List all available apps
python marker_launcher.py server           # Start the MCP API server
python marker_launcher.py convert --help   # Conversion CLI
python marker_launcher.py monitor          # Launch monitoring system
```

### 2. File Conversion
- Upload files to the `uploads/` directory.
- Use the MCP server or CLI to convert files:
  - Output is saved in `conversion_results/{filename}.md`.
- Example (via MCP):
  - Tool: `convert_file`
  - Args: `{ "file_path": "uploads/AI Agents and MCP.pdf" }`

### 3. MCP Server (Model Context Protocol)
- Exposes resources and tools for integration (e.g., with Claude Desktop)
- **Resources:**
  - `monitor://status` — Monitoring summary
  - `config://settings` — Current configuration (JSON)
  - `files://uploads` — List of uploaded files
- **Tools:**
  - `convert_file` — Convert a file to Markdown
  - `update_config` — Update configuration
  - `start_monitor`, `stop_monitor`, `plot_monitor` — Monitoring controls

#### How MCP Works
- Clients send requests for resources or tools (e.g., via Claude Desktop)
- Server responds with JSON data or performs actions (conversion, config update, etc.)
- All file paths are resolved relative to the project root
- Configuration is managed as JSON and can be updated live

### 4. Configuration Setups for Different Clients
- **Default config:** `config_fixes/optimized_config.json`
- **Claude Desktop:**
  - Uses `claude_desktop_config.json` (typically in user Library/Application Support)
  - MCP server reads config via `config://settings` and updates via `update_config` tool
- **Custom Clients:**
  - Point to the MCP server endpoint
  - Use the resource/tool URIs as described above
  - Update config by sending new JSON via the `update_config` tool

## Example: End-to-End Conversion via MCP
1. Upload a PDF to `uploads/`
2. Call the `convert_file` tool with the file path
3. Converted Markdown appears in `conversion_results/`
4. Check or update config via `config://settings` or `update_config`

## Output Locations
- **Converted files:** `conversion_results/`
- **Logs:** `marker_log.txt`, `monitoring/monitoring.log`

## Extending/Customizing
- Add new converters in `marker/converters/`
- Add new monitoring features in `unified_monitoring.py` or `monitoring/`
- Update configuration schemas in `config_fixes/optimized_config.py`

## License
See `LICENSE` for details.

---
For more details, see the original guides: `README.md`, `MONITORING_GUIDE.md`, `MONITORING_SUMMARY.md`, `COMPREHENSIVE_GUIDE.md`, `PROJECT_COMPLETION_SUMMARY.md`, `PROJECT_SUCCESS_SUMMARY.md`, `MCP_FIXES_COMPLETION.md`.