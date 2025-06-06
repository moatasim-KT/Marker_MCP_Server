import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8080/api',
  timeout: 300000, // 5 minutes for long-running conversions
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    config.params = {
      ...config.params,
      _t: Date.now(),
    };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// Health and monitoring endpoints
export const healthAPI = {
  getHealth: () => api.get('/health'),
  getMetrics: (hours = 1) => api.get(`/metrics?hours=${hours}`),
};

// Job management endpoints
export const jobsAPI = {
  getJobs: () => api.get('/jobs'),
  getJobStatus: (jobId) => api.get(`/jobs/${jobId}`),
};

// PDF conversion endpoints
export const conversionAPI = {
  convertSingle: (file, config = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('config', JSON.stringify(config));
    
    return api.post('/convert/single', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  convertBatch: (files, config = {}) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('config', JSON.stringify(config));
    
    return api.post('/convert/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  convertChunk: (file, config = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('config', JSON.stringify(config));
    
    return api.post('/convert/chunk', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};

// WebSocket connection for real-time updates
export class WebSocketManager {
  constructor() {
    this.ws = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  connect() {
    try {
      this.ws = new WebSocket('ws://localhost:8080/ws');
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.emit('message', data);
          
          // Emit specific event types
          if (data.type) {
            this.emit(data.type, data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.emit('disconnected');
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.attemptReconnect();
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in WebSocket event callback:', error);
        }
      });
    }
  }
}

// Create singleton instance
export const wsManager = new WebSocketManager();

export default api;
