/**
 * Marker MCP Dashboard - Frontend JavaScript
 * 
 * Handles WebSocket communication with the server and updates the UI
 * with real-time metrics and status information.
 */

// Global variables
let socket;
let resourceChart;
let metricsHistoryChart;
let isConnected = false;

// DOM elements
const connectionStatus = document.getElementById('connection-status');
const systemHealthEl = document.getElementById('system-health');
const activeJobsEl = document.getElementById('active-jobs');
const alertsEl = document.getElementById('alerts');

// Initialize charts
function initializeCharts() {
    // Resource Usage Chart
    const resourceCtx = document.getElementById('resource-usage-chart')?.getContext('2d');
    if (!resourceCtx) return;
    
    resourceChart = new Chart(resourceCtx, {
        type: 'doughnut',
        data: {
            labels: ['CPU', 'Memory', 'GPU'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(255, 159, 64, 0.7)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label;
                            const value = context.raw;
                            if (label === 'GPU' && window.lastGpuInfo) {
                                const gpuMB = window.lastGpuInfo.gpu_memory_mb || 0;
                                const gpuType = window.lastGpuInfo.gpu_type || 'None';
                                return `${label} (${gpuType}): ${value}% (${gpuMB.toFixed(1)}MB)`;
                            }
                            return `${label}: ${value}%`;
                        }
                    }
                }
            }
        }
    });

    // Metrics History Chart
    const historyCtx = document.getElementById('metrics-history-chart')?.getContext('2d');
    if (!historyCtx) return;
    
    metricsHistoryChart = new Chart(historyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Memory %',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'GPU Memory %',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    hidden: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Usage %'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            animation: {
                duration: 0
            },
            elements: {
                point: {
                    radius: 0
                }
            }
        }
    });
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('WebSocket connected');
            isConnected = true;
            updateConnectionStatus('connected');
        };
        
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'metrics_update') {
                    updateDashboard(data);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        socket.onclose = () => {
            console.log('WebSocket disconnected');
            isConnected = false;
            updateConnectionStatus('disconnected');
            
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 5000);
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus('error');
        };
    } catch (error) {
        console.error('Error creating WebSocket:', error);
        updateConnectionStatus('error');
        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 5000);
    }
}

// Update connection status UI
function updateConnectionStatus(status) {
    if (!connectionStatus) return;
    
    const statusMap = {
        'connected': { text: 'Connected', bg: 'bg-green-100', textColor: 'text-green-800' },
        'disconnected': { text: 'Disconnected', bg: 'bg-yellow-100', textColor: 'text-yellow-800' },
        'connecting': { text: 'Connecting...', bg: 'bg-blue-100', textColor: 'text-blue-800' },
        'error': { text: 'Connection Error', bg: 'bg-red-100', textColor: 'text-red-800' }
    };
    
    const statusInfo = statusMap[status] || statusMap['disconnected'];
    connectionStatus.textContent = statusInfo.text;
    connectionStatus.className = `inline-block px-3 py-1 rounded-full text-sm font-medium ${statusInfo.bg} ${statusInfo.textColor}`;
}

// Update dashboard with new data
function updateDashboard(data) {
    if (!data || !data.health) return;
    
    // Update system health
    updateSystemHealth(data.health);
    
    // Update resource usage chart
    updateResourceChart(data.health);
    
    // Update active jobs
    updateActiveJobs(data.active_jobs || {});
    
    // Update metrics history
    updateMetricsHistory(data.health);
}

// Update system health section
function updateSystemHealth(health) {
    if (!systemHealthEl || !health) return;
    
    const statusMap = {
        'healthy': 'bg-green-100 text-green-800',
        'warning': 'bg-yellow-100 text-yellow-800',
        'critical': 'bg-red-100 text-red-800',
        'error': 'bg-red-100 text-red-800'
    };
    
    const statusClass = statusMap[health.status?.toLowerCase()] || 'bg-gray-100 text-gray-800';
    const statusText = health.status || 'unknown';
    
    systemHealthEl.innerHTML = `
        <div class="space-y-2">
            <div class="flex justify-between items-center">
                <span class="font-medium">Status:</span>
                <span class="px-2 py-1 rounded-full text-xs ${statusClass}">
                    ${statusText.toUpperCase()}
                </span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">Memory:</span>
                <span>${health.memory_status || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">Processing:</span>
                <span>${health.processing_status || 'N/A'}</span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">Active Jobs:</span>
                <span>${health.active_jobs || 0}</span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">Queue Size:</span>
                <span>${health.queue_size || 0}</span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">GPU Type:</span>
                <span>${health.gpu_type || 'None'}</span>
            </div>
            <div class="flex justify-between">
                <span class="font-medium">GPU Device:</span>
                <span class="text-xs" title="${health.gpu_device_name || 'No GPU'}">${(health.gpu_device_name || 'No GPU').substring(0, 20)}${(health.gpu_device_name || '').length > 20 ? '...' : ''}</span>
            </div>
        </div>
    `;
    
    // Update alerts
    if (alertsEl) {
        if (health.alerts && health.alerts.length > 0) {
            alertsEl.innerHTML = health.alerts.map(alert => 
                `<div class="p-3 bg-red-50 text-red-700 rounded-md flex items-start">
                    <svg class="h-5 w-5 text-red-500 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span>${alert}</span>
                </div>`
            ).join('');
        } else {
            alertsEl.innerHTML = `
                <div class="text-center py-4 text-gray-500">
                    <svg class="mx-auto h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p class="mt-2">No active alerts</p>
                </div>
            `;
        }
    }
}

// Update resource usage chart
function updateResourceChart(health) {
    if (!resourceChart || !health) return;

    // Store GPU info globally for tooltip
    window.lastGpuInfo = {
        gpu_memory_mb: health.gpu_memory_mb || 0,
        gpu_type: health.gpu_type || 'None'
    };

    resourceChart.data.datasets[0].data = [
        health.cpu_percent || 0,
        health.memory_percent || 0,
        health.gpu_memory_percent || 0
    ];
    resourceChart.update();
}

// Update active jobs list
function updateActiveJobs(activeJobs) {
    if (!activeJobsEl) return;
    
    if (!activeJobs || Object.keys(activeJobs).length === 0) {
        activeJobsEl.innerHTML = `
            <div class="text-center py-4 text-gray-500">
                <svg class="mx-auto h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p class="mt-2">No active jobs</p>
            </div>
        `;
        return;
    }
    
    activeJobsEl.innerHTML = Object.entries(activeJobs).map(([id, job]) => {
        const progress = job.progress_percent ? Math.round(job.progress_percent) : 0;
        const startTime = job.started_at ? new Date(job.started_at * 1000).toLocaleTimeString() : 'N/A';
        const fileName = job.file_path ? job.file_path.split('/').pop() : 'Unknown file';
        const duration = job.duration_so_far ? Math.round(job.duration_so_far) : 0;
        const pageInfo = job.current_page && job.total_pages ? `${job.current_page}/${job.total_pages}` : 'N/A';
        const stage = job.processing_stage || 'Processing';
        const fileSize = job.file_size_mb ? `${job.file_size_mb.toFixed(1)}MB` : 'N/A';

        return `
            <div class="border rounded-md p-4 mb-3">
                <div class="flex justify-between items-center mb-2">
                    <span class="font-medium truncate">${job.operation || 'Conversion'}</span>
                    <span class="text-sm text-gray-500 truncate ml-2 max-w-[200px] text-right" title="${fileName}">
                        ${fileName}
                    </span>
                </div>
                <div class="text-xs text-gray-600 mb-2">
                    <div class="flex justify-between">
                        <span>Stage: ${stage}</span>
                        <span>Size: ${fileSize}</span>
                    </div>
                    <div class="flex justify-between">
                        <span>Pages: ${pageInfo}</span>
                        <span>Duration: ${duration}s</span>
                    </div>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2.5 mb-2">
                    <div class="bg-blue-600 h-2.5 rounded-full transition-all duration-300" style="width: ${progress}%"></div>
                </div>
                <div class="flex justify-between text-xs text-gray-500">
                    <span>Progress: ${progress}%</span>
                    <span>Started: ${startTime}</span>
                </div>
            </div>
        `;
    }).join('');
}

// Update metrics history chart
function updateMetricsHistory(health) {
    if (!metricsHistoryChart || !health) return;
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Add new data point
    metricsHistoryChart.data.labels.push(timeStr);
    metricsHistoryChart.data.datasets[0].data.push(health.cpu_percent || 0);
    metricsHistoryChart.data.datasets[1].data.push(health.memory_percent || 0);
    
    // Add GPU data if available
    if (health.gpu_memory_percent !== undefined) {
        if (metricsHistoryChart.data.datasets.length < 3) {
            metricsHistoryChart.data.datasets.push({
                label: 'GPU Memory %',
                data: [],
                borderColor: 'rgba(255, 159, 64, 1)',
                backgroundColor: 'rgba(255, 159, 64, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            });
        }
        metricsHistoryChart.data.datasets[2].data.push(health.gpu_memory_percent);
    }
    
    // Limit the number of data points to show
    const maxDataPoints = 20;
    if (metricsHistoryChart.data.labels.length > maxDataPoints) {
        metricsHistoryChart.data.labels.shift();
        metricsHistoryChart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }
    
    metricsHistoryChart.update();
}

// Initialize the dashboard when the DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}

function initializeDashboard() {
    initializeCharts();
    connectWebSocket();
    updateConnectionStatus('connecting');
    
    // Periodically check connection status
    setInterval(() => {
        if (socket) {
            if (socket.readyState === WebSocket.OPEN) {
                updateConnectionStatus('connected');
            } else if (socket.readyState === WebSocket.CLOSED) {
                updateConnectionStatus('disconnected');
            } else if (socket.readyState === WebSocket.CONNECTING) {
                updateConnectionStatus('connecting');
            }
        }
    }, 1000);
}
