import React, { useEffect, useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  PictureAsPdf,
  Speed,
  CheckCircle,
  Error,
  Warning,
  Memory,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

import { healthAPI, jobsAPI, wsManager } from '../utils/api';

function StatCard({ title, value, icon, color = 'primary', subtitle, loading = false }) {
  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="h6" color="textSecondary" gutterBottom>
              {title}
            </Typography>
            {loading ? (
              <CircularProgress size={24} />
            ) : (
              <>
                <Typography variant="h4" color={color}>
                  {value}
                </Typography>
                {subtitle && (
                  <Typography variant="body2" color="textSecondary">
                    {subtitle}
                  </Typography>
                )}
              </>
            )}
          </Box>
          <Box color={`${color}.main`}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

function SystemHealthCard({ health, loading }) {
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Health
          </Typography>
          <Box display="flex" justifyContent="center">
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Box mb={2}>
          <Chip 
            label={`Status: ${health?.status || 'Unknown'}`}
            color={getStatusColor(health?.status)}
            variant="outlined"
          />
        </Box>
        
        {health?.memory_status && (
          <Box mb={2}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Memory Usage
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={health.memory_status.percent || 0}
              color={health.memory_status.percent > 80 ? 'error' : 'primary'}
            />
            <Typography variant="caption" color="textSecondary">
              {health.memory_status.used_mb?.toFixed(1)} MB / {health.memory_status.total_mb?.toFixed(1)} MB
            </Typography>
          </Box>
        )}

        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip 
            size="small" 
            label={`Active Jobs: ${health?.active_jobs || 0}`}
            icon={<Speed />}
          />
          <Chip 
            size="small" 
            label={`Queue: ${health?.queue_size || 0}`}
            icon={<Memory />}
          />
        </Box>

        {health?.alerts && health.alerts.length > 0 && (
          <Box mt={2}>
            {health.alerts.map((alert, index) => (
              <Alert key={index} severity={alert.severity || 'info'} size="small">
                {alert.message}
              </Alert>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

function Dashboard() {
  const [realtimeData, setRealtimeData] = useState({});
  
  // Fetch health data
  const { data: healthData, isLoading: healthLoading, refetch: refetchHealth } = useQuery(
    'health',
    healthAPI.getHealth,
    {
      refetchInterval: 10000, // Refetch every 10 seconds
      select: (response) => response.data,
    }
  );

  // Fetch metrics data
  const { data: metricsData, isLoading: metricsLoading } = useQuery(
    'metrics',
    () => healthAPI.getMetrics(24), // Last 24 hours
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      select: (response) => response.data?.metrics,
    }
  );

  // Fetch jobs data
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery(
    'jobs',
    jobsAPI.getJobs,
    {
      refetchInterval: 5000, // Refetch every 5 seconds
      select: (response) => response.data,
    }
  );

  // WebSocket connection for real-time updates
  useEffect(() => {
    wsManager.connect();

    const handleJobUpdate = (data) => {
      setRealtimeData(prev => ({
        ...prev,
        lastJobUpdate: data,
      }));
      refetchJobs();
    };

    const handleSystemUpdate = (data) => {
      setRealtimeData(prev => ({
        ...prev,
        lastSystemUpdate: data,
      }));
      refetchHealth();
    };

    wsManager.on('job_started', handleJobUpdate);
    wsManager.on('job_completed', handleJobUpdate);
    wsManager.on('system_update', handleSystemUpdate);

    return () => {
      wsManager.off('job_started', handleJobUpdate);
      wsManager.off('job_completed', handleJobUpdate);
      wsManager.off('system_update', handleSystemUpdate);
    };
  }, [refetchJobs, refetchHealth]);

  // Calculate stats from jobs data
  const jobStats = React.useMemo(() => {
    if (!jobsData) return {};

    const totalJobs = jobsData.job_history?.length || 0;
    const activeJobs = Object.keys(jobsData.active_jobs || {}).length;
    const completedJobs = jobsData.job_history?.filter(job => job.status === 'completed').length || 0;
    const failedJobs = jobsData.job_history?.filter(job => job.status === 'failed').length || 0;

    return {
      total: totalJobs,
      active: activeJobs,
      completed: completedJobs,
      failed: failedJobs,
      successRate: totalJobs > 0 ? ((completedJobs / totalJobs) * 100).toFixed(1) : 0,
    };
  }, [jobsData]);

  // Prepare chart data
  const performanceData = React.useMemo(() => {
    if (!metricsData?.performance_history) return [];
    
    return metricsData.performance_history.slice(-24).map((item, index) => ({
      time: new Date(item.timestamp * 1000).toLocaleTimeString(),
      memory: item.memory_mb,
      cpu: item.cpu_percent,
      processing_time: item.avg_processing_time || 0,
    }));
  }, [metricsData]);

  const jobTypeData = React.useMemo(() => {
    if (!jobsData?.job_history) return [];

    const typeCounts = jobsData.job_history.reduce((acc, job) => {
      const type = job.type || 'unknown';
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(typeCounts).map(([type, count]) => ({
      type: type.replace('_', ' ').toUpperCase(),
      count,
    }));
  }, [jobsData]);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Jobs"
            value={jobStats.total}
            icon={<PictureAsPdf />}
            loading={jobsLoading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Active Jobs"
            value={jobStats.active}
            icon={<Speed />}
            color="warning"
            loading={jobsLoading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Completed"
            value={jobStats.completed}
            icon={<CheckCircle />}
            color="success"
            loading={jobsLoading}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Success Rate"
            value={`${jobStats.successRate}%`}
            icon={<CheckCircle />}
            color="success"
            subtitle={`${jobStats.failed} failed`}
            loading={jobsLoading}
          />
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={6}>
          <SystemHealthCard health={healthData} loading={healthLoading} />
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              {jobsLoading ? (
                <Box display="flex" justifyContent="center">
                  <CircularProgress />
                </Box>
              ) : (
                <Box>
                  {jobsData?.job_history?.slice(-5).reverse().map((job, index) => (
                    <Box key={job.id || index} mb={1}>
                      <Box display="flex" alignItems="center" justifyContent="space-between">
                        <Typography variant="body2">
                          {job.filename || job.filenames?.join(', ') || 'Unknown file'}
                        </Typography>
                        <Chip 
                          size="small"
                          label={job.status}
                          color={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'default'}
                        />
                      </Box>
                      <Typography variant="caption" color="textSecondary">
                        {job.started_at ? new Date(job.started_at).toLocaleString() : 'Unknown time'}
                      </Typography>
                    </Box>
                  ))}
                  {(!jobsData?.job_history || jobsData.job_history.length === 0) && (
                    <Typography variant="body2" color="textSecondary">
                      No recent activity
                    </Typography>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance (24h)
              </Typography>
              {metricsLoading ? (
                <Box display="flex" justifyContent="center" height={300}>
                  <CircularProgress />
                </Box>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="memory" stroke="#8884d8" name="Memory (MB)" />
                    <Line type="monotone" dataKey="cpu" stroke="#82ca9d" name="CPU %" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Job Types Chart */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Job Types
              </Typography>
              {jobsLoading ? (
                <Box display="flex" justifyContent="center" height={300}>
                  <CircularProgress />
                </Box>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={jobTypeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
