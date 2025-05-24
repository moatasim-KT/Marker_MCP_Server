#!/usr/bin/env python3
"""
Marker MCP Server
Exposes Marker operations via Model Context Protocol (MCP) and provides a simple web UI dashboard.
"""
import json
from fastapi import FastAPI, Response, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
from unified_monitoring import UnifiedMonitoringSystem
from config_fixes.optimized_config import load_config, save_config
import uvicorn
from pathlib import Path  # Add import for static path resolution
import os
import time
import tempfile

# Initialize MCP server
mcp = FastMCP("MarkerServer")

# Monitoring Resources and Tools
@mcp.resource("monitor://status")
def get_monitor_status() -> str:
    """Get current monitoring status summary"""
    monitor = UnifiedMonitoringSystem()
    # Use print to capture summary str
    from io import StringIO
    buf = StringIO()
    sys_stdout = __import__('sys').stdout
    try:
        __import__('sys').stdout = buf
        monitor.print_status_summary()
        return buf.getvalue()
    finally:
        __import__('sys').stdout = sys_stdout

@mcp.tool("monitor://start")
def start_monitor() -> str:
    """Start real-time monitoring"""
    monitor = UnifiedMonitoringSystem()
    monitor.start_realtime_monitoring()
    return "Monitoring started"

@mcp.tool("monitor://stop")
def stop_monitor() -> str:
    """Stop real-time monitoring"""
    monitor = UnifiedMonitoringSystem()
    monitor.stop_realtime_monitoring()
    return "Monitoring stopped"

@mcp.tool("monitor://plot")
def plot_monitor(days: int = 7) -> str:
    """Generate performance plot"""
    monitor = UnifiedMonitoringSystem()
    path = f"monitoring/plot_{days}d.png"
    monitor.generate_performance_plot(days, save_path=path)
    return path

# Configuration Resources and Tools
@mcp.resource("config://settings")
def get_config_settings() -> dict:
    """Get current configuration settings"""
    return load_config()

@mcp.tool("config://update")
def update_config_tool(config_json: str) -> str:
    """Update configuration settings"""
    config = json.loads(config_json)
    save_config(config)
    return "Configuration updated"

# Create FastAPI app and mount UI and MCP streamable HTTP
app = FastAPI()
app.mount("/mcp", mcp.streamable_http_app())

# Compute absolute path to root-level static folder and mount if it exists
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists() and static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    print(f"Note: Static directory {static_dir} not found. Web UI assets will not be available.")

# Custom endpoints to directly serve monitoring and config data
@app.get("/api/monitor/status")
def api_monitor_status():
    """Get monitoring status via direct API"""
    try:
        return get_monitor_status()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/config/settings")
def api_config_settings():
    """Get config settings via direct API"""
    try:
        return get_config_settings()
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/monitor/start")
def api_monitor_start():
    """Start monitoring via direct API"""
    try:
        result = start_monitor()
        return {"message": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/monitor/stop")
def api_monitor_stop():
    """Stop monitoring via direct API"""
    try:
        result = stop_monitor()
        return {"message": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/convert")
async def api_convert_file(file: UploadFile = File(...)):
    """Convert uploaded file to markdown"""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="File has no name")
    
    try:
        start_time = time.time()
        
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        input_path = uploads_dir / file.filename
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Prepare output path
        output_dir = Path("conversion_results")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"{input_path.stem}.md"
        output_path = output_dir / output_filename
        
        # Import and use marker conversion
        from unified_conversion import UnifiedConversionSystem
        converter = UnifiedConversionSystem()
        
        # Convert file
        result = converter.convert_single_pdf(str(input_path), str(output_path))
        
        processing_time = time.time() - start_time
        
        if result and output_path.exists():
            return {
                "filename": output_filename,
                "processing_time": f"{processing_time:.2f}s",
                "message": "Conversion successful"
            }
        else:
            return {"error": "Conversion failed"}
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/downloads/{filename}")
def download_file(filename: str):
    """Download converted file"""
    file_path = Path("conversion_results") / filename
    if file_path.exists():
        return FileResponse(str(file_path), filename=filename)
    else:
        return {"error": "File not found"}

@app.get("/")
def dashboard():
    # Simple HTML dashboard with tabs for monitoring, progress, and customizations
    html = """
<!DOCTYPE html>
<html>
<head>
  <title>Marker Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .tabs { display: flex; background: #222; }
    .tabs button { flex: 1; padding: 14px; border: none; background: #222; color: #fff; cursor: pointer; }
    .tabs button.active { background: #555; }
    .content { padding: 20px; }
  </style>
</head>
<body>
  <div class="tabs">
    <button class="active" onclick="showTab('monitor')">Monitoring</button>
    <button onclick="showTab('convert')">Convert Files</button>
    <button onclick="showTab('progress')">Progress</button>
    <button onclick="showTab('custom')">Settings</button>
  </div>
  <div class="content">
    <div id="monitor">
      <h2>Monitoring</h2>
      <pre id="monitor-status">Loading...</pre>
      <div style="margin: 10px 0;">
        <button onclick="refreshStatus()">Refresh Status</button>
        <button onclick="startMonitoring()">Start Monitoring</button>
        <button onclick="stopMonitoring()">Stop Monitoring</button>
      </div>
    </div>
    <div id="convert" style="display:none;">
      <h2>Convert Files</h2>
      <div style="border: 2px dashed #ccc; padding: 20px; margin: 10px 0; text-align: center;">
        <input type="file" id="fileInput" accept=".pdf,.docx,.pptx" style="margin: 10px;">
        <br>
        <button onclick="convertFile()" style="padding: 10px 20px; margin: 10px;">Convert to Markdown</button>
      </div>
      <div id="conversionResult" style="margin: 10px 0;"></div>
    </div>
    <div id="progress" style="display:none;">
      <h2>Progress</h2>
      <p>Progress tracking coming soon.</p>
    </div>
    <div id="custom" style="display:none;">
      <h2>Settings</h2>
      <pre id="config">Loading...</pre>
      <button onclick="loadConfig()">Reload Config</button>
    </div>
  </div>
  <script>
    function showTab(tab) {
      document.querySelectorAll('.content > div').forEach(div => div.style.display = 'none');
      document.getElementById(tab).style.display = 'block';
      document.querySelectorAll('.tabs button').forEach(btn => btn.classList.remove('active'));
      event.target.classList.add('active');
    }
    async function refreshStatus() {
      try {
        const res = await fetch('/api/monitor/status');
        const text = await res.text();
        document.getElementById('monitor-status').textContent = text;
      } catch (error) {
        document.getElementById('monitor-status').textContent = 'Error loading status: ' + error;
      }
    }
    async function loadConfig() {
      try {
        const res = await fetch('/api/config/settings');
        const cfg = await res.json();
        document.getElementById('config').textContent = JSON.stringify(cfg, null, 2);
      } catch (error) {
        document.getElementById('config').textContent = 'Error loading config: ' + error;
      }
    }
    async function startMonitoring() {
      try {
        const res = await fetch('/api/monitor/start', { method: 'POST' });
        const result = await res.json();
        alert(result.message || result.error);
        refreshStatus();
      } catch (error) {
        alert('Error starting monitoring: ' + error);
      }
    }
    async function stopMonitoring() {
      try {
        const res = await fetch('/api/monitor/stop', { method: 'POST' });
        const result = await res.json();
        alert(result.message || result.error);
        refreshStatus();
      } catch (error) {
        alert('Error stopping monitoring: ' + error);
      }
    }
    async function convertFile() {
      const fileInput = document.getElementById('fileInput');
      const resultDiv = document.getElementById('conversionResult');
      
      if (!fileInput.files[0]) {
        alert('Please select a file first');
        return;
      }
      
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      
      resultDiv.innerHTML = '<p>Converting... Please wait.</p>';
      
      try {
        const response = await fetch('/api/convert', {
          method: 'POST',
          body: formData
        });
        
        const result = await response.json();
        
        if (result.error) {
          resultDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
        } else {
          resultDiv.innerHTML = `
            <h3>Conversion Complete!</h3>
            <p><strong>Output file:</strong> <a href="/downloads/${result.filename}" target="_blank">${result.filename}</a></p>
            <p><strong>Processing time:</strong> ${result.processing_time || 'N/A'}</p>
          `;
        }
      } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
      }
    }
    document.addEventListener('DOMContentLoaded', () => { refreshStatus(); loadConfig(); });
  </script>
</body>
</html>
    """
    return Response(content=html, media_type="text/html")

# Define CLI entrypoint for Marker MCP server

def server_cli():
    """CLI entrypoint: run the MCP server and dashboard"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Replace direct uvicorn run with CLI entrypoint
if __name__ == "__main__":
    server_cli()
