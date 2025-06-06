import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
  CardContent,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Grid,
  Alert,
  LinearProgress,
  Chip,
  Divider,
} from '@mui/material';
import {
  CloudUpload,
  Settings,
  PlayArrow,
  Stop,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';

import { conversionAPI } from '../utils/api';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`conversion-tabpanel-${index}`}
      aria-labelledby={`conversion-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
    </div>
  );
}

function FileDropzone({ onDrop, accept, multiple = false, disabled = false }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple,
    disabled,
  });

  return (
    <Box
      {...getRootProps()}
      sx={{
        border: '2px dashed',
        borderColor: isDragActive ? 'primary.main' : 'grey.300',
        borderRadius: 2,
        p: 4,
        textAlign: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        backgroundColor: isDragActive ? 'primary.light' : 'transparent',
        opacity: disabled ? 0.5 : 1,
        '&:hover': {
          borderColor: disabled ? 'grey.300' : 'primary.main',
          backgroundColor: disabled ? 'transparent' : 'primary.light',
        },
      }}
    >
      <input {...getInputProps()} />
      <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
      <Typography variant="h6" gutterBottom>
        {isDragActive
          ? 'Drop the PDF files here...'
          : `Drag & drop PDF ${multiple ? 'files' : 'file'} here, or click to select`}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Only PDF files are supported
      </Typography>
    </Box>
  );
}

function ConversionConfig({ config, onChange }) {
  const handleConfigChange = (field, value) => {
    onChange({
      ...config,
      [field]: value,
    });
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          <Settings sx={{ mr: 1, verticalAlign: 'middle' }} />
          Conversion Settings
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Output Format</InputLabel>
              <Select
                value={config.output_format || 'markdown'}
                label="Output Format"
                onChange={(e) => handleConfigChange('output_format', e.target.value)}
              >
                <MenuItem value="markdown">Markdown</MenuItem>
                <MenuItem value="json">JSON</MenuItem>
                <MenuItem value="html">HTML</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Max Pages"
              type="number"
              value={config.max_pages || ''}
              onChange={(e) => handleConfigChange('max_pages', parseInt(e.target.value) || null)}
              helperText="Leave empty for all pages"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Page Range"
              value={config.page_range || ''}
              onChange={(e) => handleConfigChange('page_range', e.target.value)}
              helperText="e.g., 1-5, 10, 15-20"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Pages per Chunk"
              type="number"
              value={config.pages_per_chunk || 5}
              onChange={(e) => handleConfigChange('pages_per_chunk', parseInt(e.target.value) || 5)}
              helperText="For chunked processing only"
            />
          </Grid>
          
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.use_llm || false}
                  onChange={(e) => handleConfigChange('use_llm', e.target.checked)}
                />
              }
              label="Use LLM Enhancement"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.combine_output || true}
                  onChange={(e) => handleConfigChange('combine_output', e.target.checked)}
                />
              }
              label="Combine Output (Chunked)"
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={config.debug || false}
                  onChange={(e) => handleConfigChange('debug', e.target.checked)}
                />
              }
              label="Debug Mode"
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

function ConvertPage() {
  const [tabValue, setTabValue] = useState(0);
  const [files, setFiles] = useState([]);
  const [config, setConfig] = useState({
    output_format: 'markdown',
    use_llm: false,
    debug: false,
    pages_per_chunk: 5,
    combine_output: true,
  });
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentJob, setCurrentJob] = useState(null);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setFiles([]);
    setProgress(0);
    setCurrentJob(null);
  };

  const onDrop = useCallback((acceptedFiles) => {
    setFiles(acceptedFiles);
  }, []);

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const startConversion = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one PDF file');
      return;
    }

    setProcessing(true);
    setProgress(0);

    try {
      let response;
      
      if (tabValue === 0) {
        // Single conversion
        response = await conversionAPI.convertSingle(files[0], config);
      } else if (tabValue === 1) {
        // Batch conversion
        response = await conversionAPI.convertBatch(files, config);
      } else {
        // Chunked conversion
        response = await conversionAPI.convertChunk(files[0], config);
      }

      setCurrentJob(response.data);
      
      if (response.data.status === 'completed') {
        toast.success('Conversion completed successfully!');
        setProgress(100);
      } else if (response.data.status === 'failed') {
        toast.error('Conversion failed');
        setProgress(0);
      } else {
        toast.success('Conversion started');
        // Start polling for status updates
        pollJobStatus(response.data.job_id);
      }
    } catch (error) {
      console.error('Conversion error:', error);
      toast.error(error.response?.data?.detail || 'Conversion failed');
      setProgress(0);
    } finally {
      if (currentJob?.status === 'completed' || currentJob?.status === 'failed') {
        setProcessing(false);
      }
    }
  };

  const pollJobStatus = async (jobId) => {
    // This would be implemented with the jobs API
    // For now, we'll simulate progress
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += 10;
      setProgress(currentProgress);
      
      if (currentProgress >= 100) {
        clearInterval(interval);
        setProcessing(false);
        toast.success('Conversion completed!');
      }
    }, 1000);
  };

  const stopConversion = () => {
    setProcessing(false);
    setProgress(0);
    setCurrentJob(null);
    toast.info('Conversion stopped');
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        PDF Conversion
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Single File" />
          <Tab label="Batch Processing" />
          <Tab label="Chunked Processing" />
        </Tabs>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <TabPanel value={tabValue} index={0}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Single File Conversion
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Convert a single PDF file to the specified output format.
              </Typography>
              
              <FileDropzone
                onDrop={onDrop}
                multiple={false}
                disabled={processing}
              />
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Batch Processing
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Convert multiple PDF files simultaneously for increased efficiency.
              </Typography>
              
              <FileDropzone
                onDrop={onDrop}
                multiple={true}
                disabled={processing}
              />
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Chunked Processing
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Process large PDF files in smaller chunks to handle memory constraints and improve reliability.
              </Typography>
              
              <FileDropzone
                onDrop={onDrop}
                multiple={false}
                disabled={processing}
              />
            </Box>
          </TabPanel>

          {/* File List */}
          {files.length > 0 && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Selected Files
                </Typography>
                {files.map((file, index) => (
                  <Box key={index} sx={{ mb: 1 }}>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography variant="body1">{file.name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </Typography>
                      </Box>
                      {!processing && (
                        <Button
                          size="small"
                          color="error"
                          onClick={() => removeFile(index)}
                        >
                          Remove
                        </Button>
                      )}
                    </Box>
                    {index < files.length - 1 && <Divider sx={{ mt: 1 }} />}
                  </Box>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Progress */}
          {processing && (
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processing...
                </Typography>
                <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  {progress}% Complete
                </Typography>
                
                {currentJob && (
                  <Box mt={2}>
                    <Chip 
                      label={`Job ID: ${currentJob.job_id}`}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <Chip 
                      label={currentJob.status}
                      color={currentJob.status === 'completed' ? 'success' : 'primary'}
                      size="small"
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* Control Buttons */}
          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={startConversion}
              disabled={processing || files.length === 0}
            >
              Start Conversion
            </Button>
            
            {processing && (
              <Button
                variant="outlined"
                size="large"
                startIcon={<Stop />}
                onClick={stopConversion}
                color="error"
              >
                Stop
              </Button>
            )}
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <ConversionConfig config={config} onChange={setConfig} />
          
          {/* Tips */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tips & Best Practices
              </Typography>
              
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  For large files (>50MB), use chunked processing to avoid memory issues.
                </Typography>
              </Alert>
              
              <Alert severity="tip" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  Enable LLM enhancement for better text extraction and formatting.
                </Typography>
              </Alert>
              
              <Alert severity="warning">
                <Typography variant="body2">
                  Processing time varies based on file size, complexity, and selected options.
                </Typography>
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ConvertPage;
