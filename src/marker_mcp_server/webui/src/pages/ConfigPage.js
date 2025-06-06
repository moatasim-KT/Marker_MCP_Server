import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Switch,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from '@mui/material';
import {
  ExpandMore,
  Save,
  RestoreFromTrash,
  Add,
  Delete,
  Edit,
  Visibility,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { api } from '../utils/api';

function ConfigPage() {
  const [config, setConfig] = useState({});
  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [presetDialogOpen, setPresetDialogOpen] = useState(false);
  const [newPresetName, setNewPresetName] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [unsavedChanges, setUnsavedChanges] = useState(false);

  const queryClient = useQueryClient();

  // Fetch current configuration
  const { data: configData, isLoading } = useQuery(
    'server-config',
    () => api.get('/api/config'),
    {
      onSuccess: (data) => {
        setConfig(data.data);
      }
    }
  );

  // Fetch configuration presets
  const { data: presetsData } = useQuery(
    'config-presets',
    () => api.get('/api/config/presets')
  );

  useEffect(() => {
    if (presetsData?.data) {
      setPresets(presetsData.data);
    }
  }, [presetsData]);

  // Save configuration mutation
  const saveConfigMutation = useMutation(
    (newConfig) => api.post('/api/config', newConfig),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Configuration saved successfully!', severity: 'success' });
        setUnsavedChanges(false);
        queryClient.invalidateQueries('server-config');
      },
      onError: (error) => {
        setSnackbar({ open: true, message: `Error saving configuration: ${error.message}`, severity: 'error' });
      }
    }
  );

  // Reset configuration mutation
  const resetConfigMutation = useMutation(
    () => api.post('/api/config/reset'),
    {
      onSuccess: (data) => {
        setConfig(data.data);
        setSnackbar({ open: true, message: 'Configuration reset to defaults!', severity: 'info' });
        setUnsavedChanges(false);
      }
    }
  );

  // Save preset mutation
  const savePresetMutation = useMutation(
    ({ name, config }) => api.post('/api/config/presets', { name, config }),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Preset saved successfully!', severity: 'success' });
        queryClient.invalidateQueries('config-presets');
        setPresetDialogOpen(false);
        setNewPresetName('');
      }
    }
  );

  // Load preset mutation
  const loadPresetMutation = useMutation(
    (presetName) => api.get(`/api/config/presets/${presetName}`),
    {
      onSuccess: (data) => {
        setConfig(data.data.config);
        setUnsavedChanges(true);
        setSnackbar({ open: true, message: 'Preset loaded successfully!', severity: 'success' });
      }
    }
  );

  const handleConfigChange = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    setUnsavedChanges(true);
  };

  const handleSaveConfig = () => {
    saveConfigMutation.mutate(config);
  };

  const handleResetConfig = () => {
    if (window.confirm('Are you sure you want to reset all configuration to defaults? This action cannot be undone.')) {
      resetConfigMutation.mutate();
    }
  };

  const handleSavePreset = () => {
    if (newPresetName.trim()) {
      savePresetMutation.mutate({ name: newPresetName.trim(), config });
    }
  };

  const handleLoadPreset = (presetName) => {
    loadPresetMutation.mutate(presetName);
  };

  const renderSlider = (label, value, min, max, step = 1, section, key) => (
    <Box sx={{ mb: 3 }}>
      <Typography gutterBottom>{label}</Typography>
      <Slider
        value={value || min}
        min={min}
        max={max}
        step={step}
        onChange={(e, newValue) => handleConfigChange(section, key, newValue)}
        valueLabelDisplay="auto"
        marks={[
          { value: min, label: min },
          { value: max, label: max }
        ]}
      />
    </Box>
  );

  const renderTextField = (label, value, section, key, type = 'text') => (
    <TextField
      fullWidth
      label={label}
      value={value || ''}
      type={type}
      onChange={(e) => handleConfigChange(section, key, e.target.value)}
      sx={{ mb: 2 }}
    />
  );

  const renderSwitch = (label, value, section, key) => (
    <FormControlLabel
      control={
        <Switch
          checked={value || false}
          onChange={(e) => handleConfigChange(section, key, e.target.checked)}
        />
      }
      label={label}
      sx={{ mb: 2 }}
    />
  );

  const renderSelect = (label, value, options, section, key) => (
    <FormControl fullWidth sx={{ mb: 2 }}>
      <InputLabel>{label}</InputLabel>
      <Select
        value={value || ''}
        label={label}
        onChange={(e) => handleConfigChange(section, key, e.target.value)}
      >
        {options.map(option => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );

  if (isLoading) {
    return <Typography>Loading configuration...</Typography>;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Server Configuration
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {unsavedChanges && (
            <Chip label="Unsaved Changes" color="warning" size="small" />
          )}
          <Button
            variant="outlined"
            startIcon={<Add />}
            onClick={() => setPresetDialogOpen(true)}
          >
            Save Preset
          </Button>
          <Button
            variant="outlined"
            color="warning"
            startIcon={<RestoreFromTrash />}
            onClick={handleResetConfig}
          >
            Reset to Defaults
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSaveConfig}
            disabled={saveConfigMutation.isLoading || !unsavedChanges}
          >
            Save Configuration
          </Button>
        </Box>
      </Box>

      {/* Configuration Presets */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configuration Presets
          </Typography>
          <Grid container spacing={2}>
            {presets.map((preset) => (
              <Grid item xs={12} sm={6} md={4} key={preset.name}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {preset.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" gutterBottom>
                      Created: {new Date(preset.created_at).toLocaleDateString()}
                    </Typography>
                    <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleLoadPreset(preset.name)}
                      >
                        Load
                      </Button>
                      <IconButton size="small">
                        <Visibility />
                      </IconButton>
                      <IconButton size="small" color="error">
                        <Delete />
                      </IconButton>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Server Settings */}
        <Grid item xs={12} md={6}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Server Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderTextField('Host', config.server?.host, 'server', 'host')}
              {renderTextField('Port', config.server?.port, 'server', 'port', 'number')}
              {renderSwitch('Debug Mode', config.server?.debug, 'server', 'debug')}
              {renderTextField('Log Level', config.server?.log_level, 'server', 'log_level')}
              {renderTextField('Max Workers', config.server?.max_workers, 'server', 'max_workers', 'number')}
              {renderTextField('Request Timeout (seconds)', config.server?.request_timeout, 'server', 'request_timeout', 'number')}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* PDF Processing Settings */}
        <Grid item xs={12} md={6}>
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">PDF Processing</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderSelect(
                'OCR Engine',
                config.pdf?.ocr_engine,
                [
                  { value: 'tesseract', label: 'Tesseract' },
                  { value: 'surya', label: 'Surya' },
                  { value: 'auto', label: 'Auto' }
                ],
                'pdf',
                'ocr_engine'
              )}
              {renderSelect(
                'Output Format',
                config.pdf?.output_format,
                [
                  { value: 'markdown', label: 'Markdown' },
                  { value: 'html', label: 'HTML' },
                  { value: 'json', label: 'JSON' }
                ],
                'pdf',
                'output_format'
              )}
              {renderSlider('Max Pages', config.pdf?.max_pages, 1, 1000, 1, 'pdf', 'max_pages')}
              {renderSlider('DPI', config.pdf?.dpi, 72, 600, 1, 'pdf', 'dpi')}
              {renderSwitch('Extract Images', config.pdf?.extract_images, 'pdf', 'extract_images')}
              {renderSwitch('Extract Tables', config.pdf?.extract_tables, 'pdf', 'extract_tables')}
              {renderTextField('Languages', config.pdf?.languages, 'pdf', 'languages')}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Performance Settings */}
        <Grid item xs={12} md={6}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Performance Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderSlider('Batch Size', config.performance?.batch_size, 1, 50, 1, 'performance', 'batch_size')}
              {renderSlider('Max Concurrent Jobs', config.performance?.max_concurrent_jobs, 1, 20, 1, 'performance', 'max_concurrent_jobs')}
              {renderSlider('Memory Limit (MB)', config.performance?.memory_limit, 512, 8192, 64, 'performance', 'memory_limit')}
              {renderTextField('Temp Directory', config.performance?.temp_dir, 'performance', 'temp_dir')}
              {renderSwitch('Enable Caching', config.performance?.enable_caching, 'performance', 'enable_caching')}
              {renderTextField('Cache TTL (seconds)', config.performance?.cache_ttl, 'performance', 'cache_ttl', 'number')}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Security Settings */}
        <Grid item xs={12} md={6}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Security Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderSwitch('Enable Authentication', config.security?.enable_auth, 'security', 'enable_auth')}
              {renderTextField('API Key', config.security?.api_key, 'security', 'api_key', 'password')}
              {renderSlider('Rate Limit (requests/minute)', config.security?.rate_limit, 10, 1000, 10, 'security', 'rate_limit')}
              {renderSlider('Max File Size (MB)', config.security?.max_file_size, 1, 500, 1, 'security', 'max_file_size')}
              {renderSwitch('Allow CORS', config.security?.allow_cors, 'security', 'allow_cors')}
              {renderTextField('Allowed Origins', config.security?.allowed_origins, 'security', 'allowed_origins')}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Storage Settings */}
        <Grid item xs={12} md={6}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Storage Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderTextField('Input Directory', config.storage?.input_dir, 'storage', 'input_dir')}
              {renderTextField('Output Directory', config.storage?.output_dir, 'storage', 'output_dir')}
              {renderTextField('Cache Directory', config.storage?.cache_dir, 'storage', 'cache_dir')}
              {renderSwitch('Auto Cleanup', config.storage?.auto_cleanup, 'storage', 'auto_cleanup')}
              {renderTextField('Cleanup After (hours)', config.storage?.cleanup_after, 'storage', 'cleanup_after', 'number')}
              {renderSwitch('Compress Outputs', config.storage?.compress_outputs, 'storage', 'compress_outputs')}
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Monitoring Settings */}
        <Grid item xs={12} md={6}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Monitoring Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {renderSwitch('Enable Metrics', config.monitoring?.enable_metrics, 'monitoring', 'enable_metrics')}
              {renderSwitch('Enable Health Checks', config.monitoring?.enable_health_checks, 'monitoring', 'enable_health_checks')}
              {renderTextField('Metrics Retention (days)', config.monitoring?.metrics_retention, 'monitoring', 'metrics_retention', 'number')}
              {renderSwitch('Enable Alerts', config.monitoring?.enable_alerts, 'monitoring', 'enable_alerts')}
              {renderTextField('Alert Email', config.monitoring?.alert_email, 'monitoring', 'alert_email', 'email')}
              {renderTextField('Webhook URL', config.monitoring?.webhook_url, 'monitoring', 'webhook_url')}
            </AccordionDetails>
          </Accordion>
        </Grid>
      </Grid>

      {/* Current Configuration Summary */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Current Configuration Summary
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Section</TableCell>
                  <TableCell>Settings</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Server</TableCell>
                  <TableCell>
                    {config.server?.host}:{config.server?.port} | 
                    Workers: {config.server?.max_workers} | 
                    Debug: {config.server?.debug ? 'On' : 'Off'}
                  </TableCell>
                  <TableCell>
                    <Chip label="Active" color="success" size="small" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>PDF Processing</TableCell>
                  <TableCell>
                    Engine: {config.pdf?.ocr_engine} | 
                    Format: {config.pdf?.output_format} | 
                    Max Pages: {config.pdf?.max_pages}
                  </TableCell>
                  <TableCell>
                    <Chip label="Configured" color="primary" size="small" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Performance</TableCell>
                  <TableCell>
                    Batch: {config.performance?.batch_size} | 
                    Concurrent: {config.performance?.max_concurrent_jobs} | 
                    Memory: {config.performance?.memory_limit}MB
                  </TableCell>
                  <TableCell>
                    <Chip label="Optimized" color="info" size="small" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Security</TableCell>
                  <TableCell>
                    Auth: {config.security?.enable_auth ? 'Enabled' : 'Disabled'} | 
                    Rate Limit: {config.security?.rate_limit}/min | 
                    Max File: {config.security?.max_file_size}MB
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={config.security?.enable_auth ? 'Secured' : 'Open'} 
                      color={config.security?.enable_auth ? 'success' : 'warning'} 
                      size="small" 
                    />
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Save Preset Dialog */}
      <Dialog open={presetDialogOpen} onClose={() => setPresetDialogOpen(false)}>
        <DialogTitle>Save Configuration Preset</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Preset Name"
            fullWidth
            variant="outlined"
            value={newPresetName}
            onChange={(e) => setNewPresetName(e.target.value)}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPresetDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleSavePreset}
            variant="contained"
            disabled={!newPresetName.trim() || savePresetMutation.isLoading}
          >
            Save Preset
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default ConfigPage;
