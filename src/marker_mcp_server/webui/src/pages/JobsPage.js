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
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Pagination,
  LinearProgress,
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Pause,
  Stop,
  Delete,
  Refresh,
  Download,
  Visibility,
  Cancel,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { api, wsManager } from '../utils/api';

function JobsPage() {
  const [tabValue, setTabValue] = useState(0);
  const [selectedJob, setSelectedJob] = useState(null);
  const [jobDetailsOpen, setJobDetailsOpen] = useState(false);
  const [statusFilter, setStatusFilter] = useState('all');
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [jobs, setJobs] = useState([]);

  const queryClient = useQueryClient();

  // Fetch jobs based on tab
  const getJobsEndpoint = () => {
    switch (tabValue) {
      case 0: return '/api/jobs/active';
      case 1: return '/api/jobs/completed';
      case 2: return '/api/jobs/failed';
      default: return '/api/jobs/all';
    }
  };

  const { data: jobsData, isLoading } = useQuery(
    ['jobs', tabValue, statusFilter, page],
    () => api.get(`${getJobsEndpoint()}?status=${statusFilter}&page=${page}&limit=${pageSize}`),
    { refetchInterval: 2000 }
  );

  // Job queue statistics
  const { data: queueStats } = useQuery(
    'queue-stats',
    () => api.get('/api/jobs/stats'),
    { refetchInterval: 5000 }
  );

  // WebSocket for real-time job updates
  useEffect(() => {
    const handleJobUpdate = (data) => {
      setJobs(prev => {
        const index = prev.findIndex(job => job.id === data.id);
        if (index >= 0) {
          const updated = [...prev];
          updated[index] = { ...updated[index], ...data };
          return updated;
        }
        return [data, ...prev];
      });
      queryClient.invalidateQueries('jobs');
    };

    wsManager.subscribe('job_update', handleJobUpdate);
    return () => wsManager.unsubscribe('job_update', handleJobUpdate);
  }, [queryClient]);

  // Mutations for job actions
  const cancelJobMutation = useMutation(
    (jobId) => api.post(`/api/jobs/${jobId}/cancel`),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('jobs');
      },
    }
  );

  const retryJobMutation = useMutation(
    (jobId) => api.post(`/api/jobs/${jobId}/retry`),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('jobs');
      },
    }
  );

  const deleteJobMutation = useMutation(
    (jobId) => api.delete(`/api/jobs/${jobId}`),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('jobs');
      },
    }
  );

  const clearCompletedMutation = useMutation(
    () => api.post('/api/jobs/clear-completed'),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('jobs');
      },
    }
  );

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setPage(1);
  };

  const handleJobAction = (action, jobId) => {
    switch (action) {
      case 'cancel':
        cancelJobMutation.mutate(jobId);
        break;
      case 'retry':
        retryJobMutation.mutate(jobId);
        break;
      case 'delete':
        deleteJobMutation.mutate(jobId);
        break;
      default:
        break;
    }
  };

  const handleViewDetails = (job) => {
    setSelectedJob(job);
    setJobDetailsOpen(true);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      case 'pending': return 'default';
      default: return 'default';
    }
  };

  const getJobTypeIcon = (type) => {
    switch (type) {
      case 'pdf_conversion': return '📄';
      case 'batch_conversion': return '📁';
      case 'chunk_conversion': return '🔧';
      default: return '⚙️';
    }
  };

  const formatDuration = (startTime, endTime) => {
    if (!startTime) return 'N/A';
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const duration = Math.round((end - start) / 1000);
    return `${duration}s`;
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Job Management
      </Typography>

      {/* Queue Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Jobs
              </Typography>
              <Typography variant="h4" color="primary">
                {queueStats?.data?.active || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pending Jobs
              </Typography>
              <Typography variant="h4" color="warning.main">
                {queueStats?.data?.pending || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed Today
              </Typography>
              <Typography variant="h4" color="success.main">
                {queueStats?.data?.completedToday || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Failed Jobs
              </Typography>
              <Typography variant="h4" color="error.main">
                {queueStats?.data?.failed || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, alignItems: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Status Filter</InputLabel>
          <Select
            value={statusFilter}
            label="Status Filter"
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="running">Running</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="completed">Completed</MenuItem>
            <MenuItem value="failed">Failed</MenuItem>
            <MenuItem value="cancelled">Cancelled</MenuItem>
          </Select>
        </FormControl>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => queryClient.invalidateQueries('jobs')}
        >
          Refresh
        </Button>
        <Button
          variant="outlined"
          color="warning"
          onClick={() => clearCompletedMutation.mutate()}
          disabled={clearCompletedMutation.isLoading}
        >
          Clear Completed
        </Button>
      </Box>

      {/* Job Tabs */}
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Active Jobs" />
        <Tab label="Completed Jobs" />
        <Tab label="Failed Jobs" />
        <Tab label="All Jobs" />
      </Tabs>

      {/* Jobs Table */}
      <Card>
        <CardContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Job ID</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>File/Folder</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Progress</TableCell>
                  <TableCell>Duration</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={7}>
                      <LinearProgress />
                    </TableCell>
                  </TableRow>
                ) : (
                  jobsData?.data?.jobs?.map((job) => (
                    <TableRow key={job.id}>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {job.id}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <span>{getJobTypeIcon(job.type)}</span>
                          <Typography variant="body2">
                            {job.type.replace('_', ' ')}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                          {job.input_path || job.input_folder}
                        </Typography>
                        {job.file_size && (
                          <Typography variant="caption" color="textSecondary">
                            {formatFileSize(job.file_size)}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={job.status}
                          color={getStatusColor(job.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ minWidth: 100 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress || 0}
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="caption">
                            {job.progress || 0}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {formatDuration(job.start_time, job.end_time)}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <IconButton
                            size="small"
                            onClick={() => handleViewDetails(job)}
                          >
                            <Visibility />
                          </IconButton>
                          {job.status === 'running' && (
                            <IconButton
                              size="small"
                              color="warning"
                              onClick={() => handleJobAction('cancel', job.id)}
                            >
                              <Cancel />
                            </IconButton>
                          )}
                          {job.status === 'failed' && (
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleJobAction('retry', job.id)}
                            >
                              <PlayArrow />
                            </IconButton>
                          )}
                          {(['completed', 'failed', 'cancelled'].includes(job.status)) && (
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleJobAction('delete', job.id)}
                            >
                              <Delete />
                            </IconButton>
                          )}
                          {job.output_path && job.status === 'completed' && (
                            <IconButton
                              size="small"
                              color="success"
                              onClick={() => window.open(`/api/jobs/${job.id}/download`, '_blank')}
                            >
                              <Download />
                            </IconButton>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Pagination */}
          {jobsData?.data?.total > pageSize && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
              <Pagination
                count={Math.ceil(jobsData.data.total / pageSize)}
                page={page}
                onChange={(event, value) => setPage(value)}
                color="primary"
              />
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Job Details Dialog */}
      <Dialog
        open={jobDetailsOpen}
        onClose={() => setJobDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Job Details - {selectedJob?.id}
        </DialogTitle>
        <DialogContent>
          {selectedJob && (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Status
                  </Typography>
                  <Chip
                    label={selectedJob.status}
                    color={getStatusColor(selectedJob.status)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Type
                  </Typography>
                  <Typography variant="body2">
                    {selectedJob.type.replace('_', ' ')}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Input Path
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedJob.input_path || selectedJob.input_folder}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Output Path
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                    {selectedJob.output_path || selectedJob.output_folder || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Progress
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={selectedJob.progress || 0}
                      sx={{ flexGrow: 1 }}
                    />
                    <Typography variant="body2">
                      {selectedJob.progress || 0}%
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Duration
                  </Typography>
                  <Typography variant="body2">
                    {formatDuration(selectedJob.start_time, selectedJob.end_time)}
                  </Typography>
                </Grid>
              </Grid>

              {/* Configuration */}
              {selectedJob.config && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">Configuration</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <pre style={{ fontSize: '0.875rem', overflow: 'auto' }}>
                      {JSON.stringify(selectedJob.config, null, 2)}
                    </pre>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Error Details */}
              {selectedJob.error && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1" color="error">
                      Error Details
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" color="error" sx={{ fontFamily: 'monospace' }}>
                      {selectedJob.error}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Logs */}
              {selectedJob.logs && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">Logs</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box
                      sx={{
                        maxHeight: 300,
                        overflow: 'auto',
                        backgroundColor: 'grey.100',
                        p: 1,
                        borderRadius: 1,
                      }}
                    >
                      <pre style={{ fontSize: '0.75rem', margin: 0 }}>
                        {selectedJob.logs}
                      </pre>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setJobDetailsOpen(false)}>
            Close
          </Button>
          {selectedJob?.output_path && selectedJob?.status === 'completed' && (
            <Button
              variant="contained"
              startIcon={<Download />}
              onClick={() => window.open(`/api/jobs/${selectedJob.id}/download`, '_blank')}
            >
              Download Result
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default JobsPage;
