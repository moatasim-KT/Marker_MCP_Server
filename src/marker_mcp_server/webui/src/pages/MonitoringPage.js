import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
} from '@mui/lab';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
} from 'recharts';
import { useQuery } from 'react-query';
import { api, wsManager } from '../utils/api';

function MonitoringPage() {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('1h');
  const [metrics, setMetrics] = useState([]);
  const [systemLogs, setSystemLogs] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);

  // Fetch system metrics
  const { data: systemHealth } = useQuery(
    'system-health',
    () => api.get('/api/system/health'),
    { refetchInterval: 5000 }
  );

  // Fetch performance metrics
  const { data: performanceMetrics } = useQuery(
    ['performance-metrics', timeRange],
    () => api.get(`/api/system/metrics?range=${timeRange}`),
    { refetchInterval: 10000 }
  );

  // Fetch system logs
  const { data: logs } = useQuery(
    'system-logs',
    () => api.get('/api/system/logs?limit=100'),
    { refetchInterval: 5000 }
  );

  // WebSocket for real-time updates
  useEffect(() => {
    const handleMetricsUpdate = (data) => {
      setMetrics(prev => [...prev.slice(-49), data].slice(-50));
    };

    const handleLogUpdate = (data) => {
      setSystemLogs(prev => [data, ...prev.slice(0, 99)]);
    };

    wsManager.subscribe('metrics', handleMetricsUpdate);
    wsManager.subscribe('logs', handleLogUpdate);

    return () => {
      wsManager.unsubscribe('metrics', handleMetricsUpdate);
      wsManager.unsubscribe('logs', handleLogUpdate);
    };
  }, []);

  // Generate sample performance data
  useEffect(() => {
    if (performanceMetrics?.data) {
      const data = Array.from({ length: 50 }, (_, i) => ({
        time: new Date(Date.now() - (49 - i) * 60000).toLocaleTimeString(),
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        conversions: Math.floor(Math.random() * 10),
      }));
      setPerformanceData(data);
    }
  }, [performanceMetrics]);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getLogSeverityColor = (severity) => {
    switch (severity) {
      case 'error': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      case 'debug': return 'default';
      default: return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Monitoring
      </Typography>

      <Box sx={{ mb: 3 }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="15m">15 minutes</MenuItem>
            <MenuItem value="1h">1 hour</MenuItem>
            <MenuItem value="6h">6 hours</MenuItem>
            <MenuItem value="24h">24 hours</MenuItem>
          </Select>
        </FormControl>
        <Button
          variant="outlined"
          sx={{ ml: 2 }}
          onClick={() => window.location.reload()}
        >
          Refresh
        </Button>
      </Box>

      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="System Health" />
        <Tab label="Performance Metrics" />
        <Tab label="System Logs" />
        <Tab label="Resource Usage" />
      </Tabs>

      {/* System Health Tab */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Current System Status
                </Typography>
                <Grid container spacing={2}>
                  {systemHealth?.data?.components?.map((component, index) => (
                    <Grid item xs={12} sm={6} md={3} key={index}>
                      <Box
                        sx={{
                          p: 2,
                          border: 1,
                          borderColor: 'divider',
                          borderRadius: 1,
                          textAlign: 'center',
                        }}
                      >
                        <Typography variant="subtitle2" color="textSecondary">
                          {component.name}
                        </Typography>
                        <Chip
                          label={component.status}
                          color={getStatusColor(component.status)}
                          size="small"
                          sx={{ mt: 1 }}
                        />
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          {component.message}
                        </Typography>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Alerts
                </Typography>
                <Timeline>
                  {systemHealth?.data?.alerts?.slice(0, 5).map((alert, index) => (
                    <TimelineItem key={index}>
                      <TimelineSeparator>
                        <TimelineDot color={getStatusColor(alert.severity)} />
                        {index < 4 && <TimelineConnector />}
                      </TimelineSeparator>
                      <TimelineContent>
                        <Typography variant="subtitle2">
                          {alert.message}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {new Date(alert.timestamp).toLocaleString()}
                        </Typography>
                      </TimelineContent>
                    </TimelineItem>
                  ))}
                </Timeline>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Service Dependencies
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Service</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Response Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {systemHealth?.data?.dependencies?.map((dep, index) => (
                        <TableRow key={index}>
                          <TableCell>{dep.name}</TableCell>
                          <TableCell>
                            <Chip
                              label={dep.status}
                              color={getStatusColor(dep.status)}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{dep.responseTime}ms</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Performance Metrics Tab */}
      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Real-time Performance Metrics
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="cpu"
                      stroke="#8884d8"
                      name="CPU (%)"
                    />
                    <Line
                      type="monotone"
                      dataKey="memory"
                      stroke="#82ca9d"
                      name="Memory (%)"
                    />
                    <Line
                      type="monotone"
                      dataKey="disk"
                      stroke="#ffc658"
                      name="Disk (%)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Conversion Rate
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="conversions"
                      stroke="#8884d8"
                      fill="#8884d8"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Average Response Times
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={performanceMetrics?.data?.averageResponseTimes || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="endpoint" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="responseTime" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* System Logs Tab */}
      {tabValue === 2 && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Logs
            </Typography>
            <TableContainer sx={{ maxHeight: 600 }}>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Level</TableCell>
                    <TableCell>Component</TableCell>
                    <TableCell>Message</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(logs?.data || systemLogs).map((log, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {new Date(log.timestamp).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={log.level}
                          color={getLogSeverityColor(log.level)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{log.component}</TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {log.message}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Resource Usage Tab */}
      {tabValue === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  CPU Usage
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h4" color="primary">
                    {systemHealth?.data?.cpu?.usage || 0}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth?.data?.cpu?.usage || 0}
                    sx={{ mt: 1 }}
                  />
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Cores: {systemHealth?.data?.cpu?.cores || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Memory Usage
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h4" color="primary">
                    {systemHealth?.data?.memory?.usage || 0}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth?.data?.memory?.usage || 0}
                    sx={{ mt: 1 }}
                  />
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Used: {systemHealth?.data?.memory?.used || 'N/A'} / {systemHealth?.data?.memory?.total || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Disk Usage
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h4" color="primary">
                    {systemHealth?.data?.disk?.usage || 0}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={systemHealth?.data?.disk?.usage || 0}
                    sx={{ mt: 1 }}
                  />
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Free: {systemHealth?.data?.disk?.free || 'N/A'} / {systemHealth?.data?.disk?.total || 'N/A'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Process Information
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Process</TableCell>
                        <TableCell>PID</TableCell>
                        <TableCell>CPU %</TableCell>
                        <TableCell>Memory %</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {systemHealth?.data?.processes?.map((process, index) => (
                        <TableRow key={index}>
                          <TableCell>{process.name}</TableCell>
                          <TableCell>{process.pid}</TableCell>
                          <TableCell>{process.cpu}%</TableCell>
                          <TableCell>{process.memory}%</TableCell>
                          <TableCell>
                            <Chip
                              label={process.status}
                              color={process.status === 'running' ? 'success' : 'default'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}

export default MonitoringPage;
