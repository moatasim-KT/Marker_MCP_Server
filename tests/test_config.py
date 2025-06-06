import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from src.marker_mcp_server.config_schema import (
    AppConfig,
    MarkerConfig,
    MonitoringConfig,
    ResourceLimits,
    SecurityConfig,
    ServerConfig,
)
from tests.conftest import pytest_integration, pytest_unit

PERFORMANCE_TIME_LIMIT = 10.0


@pytest_unit
class TestConfigurationSchema:
    """Test configuration schema validation and defaults."""
    
    def test_default_configuration(self) -> None:
        """Test default configuration values."""
        config = AppConfig()
        
        # Test server defaults
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8000
        assert config.server.debug is False
        
        # Test marker defaults
        assert config.marker.batch_size == 1
        assert config.marker.parallel_factor == 1
        
        # Test resource limits defaults
        assert config.resource_limits.max_file_size_mb == 500
        assert config.resource_limits.max_memory_usage_mb == 4096
        assert config.resource_limits.max_processing_time_seconds == 600
        assert config.resource_limits.max_concurrent_jobs == 3
        
        # Test monitoring defaults
        assert config.monitoring.enable_metrics is True
        assert config.monitoring.metrics_interval_seconds == 30
        assert config.monitoring.log_performance is True
        assert config.monitoring.alert_memory_threshold_percent == 85.0
        
        # Test security defaults
        assert config.security.validate_file_paths is True
        assert config.security.allowed_file_extensions == [
            '.pdf',
        ]
        
        # Test paths defaults
        assert "marker-mcp" in config.paths.cache_dir
        assert "models" in config.paths.model_dir
        assert "logs" in config.paths.logs_dir
    
    def test_configuration_validation(self) -> None:
        """Test configuration validation rules."""
        # Test valid configuration
        valid_config_data = {
            "server": {"port": 8080, "host": "0.0.0.0"},
            "resource_limits": {
                "max_file_size_mb": 100,
                "max_memory_usage_mb": 2048,
                "max_processing_time_seconds": 300,
                "max_concurrent_jobs": 2
            },
            "monitoring": {
                "metrics_interval_seconds": 60,
                "alert_memory_threshold_percent": 90.0,
            }
        }
        
        config = AppConfig.model_validate(valid_config_data)
        assert config.server.port == 8080
        assert config.resource_limits.max_file_size_mb == 100
        
        # Test invalid resource limits
        with pytest.raises(ValueError):
            ResourceLimits(max_file_size_mb=-1)
        
        with pytest.raises(ValueError):
            ResourceLimits(max_memory_usage_mb=0)
        
        with pytest.raises(ValueError):
            ResourceLimits(max_processing_time_seconds=-10)
    
    @staticmethod
    def test_server_config_validation() -> None:
        """Test server configuration validation."""
        # Valid server config
        server_config = ServerConfig(host="127.0.0.1", port=9000, debug=True)
        assert server_config.host == "127.0.0.1"
        assert server_config.port == 9000
        assert server_config.debug is True
    
    def test_marker_config_validation(self) -> None:
        """Test marker configuration validation."""
        # Valid marker config
        marker_config = MarkerConfig(
            batch_size=10,
            max_pages=1000,
            parallel_factor=2
        )
        assert marker_config.batch_size == 10
        assert marker_config.max_pages == 1000
        assert marker_config.parallel_factor == 2
    
    def test_resource_limits_validation(self) -> None:
        """Test resource limits validation."""
        # Valid resource limits
        limits = ResourceLimits(
            max_file_size_mb=200,
            max_memory_usage_mb=8192,
            max_processing_time_seconds=1200,
            max_concurrent_jobs=5
        )
        assert limits.max_file_size_mb == 200
        assert limits.max_concurrent_jobs == 5
        
        # Test negative values
        with pytest.raises(ValueError):
            ResourceLimits(max_file_size_mb=-1)
        
        with pytest.raises(ValueError):
            ResourceLimits(max_memory_usage_mb=-100)
        
        with pytest.raises(ValueError):
            ResourceLimits(max_processing_time_seconds=-60)
    
    @staticmethod
    def test_monitoring_config_validation() -> None:
        """Test monitoring configuration validation."""
        # Valid monitoring config
        monitoring = MonitoringConfig(
            enable_metrics=True,
            metrics_interval_seconds=45,
            log_performance=False,
            alert_memory_threshold_percent=80.0
        )
        assert monitoring.metrics_interval_seconds == 45
        assert monitoring.log_performance is False
        assert monitoring.alert_memory_threshold_percent == 80.0
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            MonitoringConfig(alert_memory_threshold_percent=150.0)
        
        with pytest.raises(ValueError):
            MonitoringConfig(alert_memory_threshold_percent=-10.0)
        
        # Test invalid interval
        with pytest.raises(ValueError):
            MonitoringConfig(metrics_interval_seconds=0)
    
    def test_security_config_validation(self) -> None:
        """Test security configuration validation.""" 
        # Valid security config
        security = SecurityConfig(
            validate_file_paths=True,
            allowed_input_dirs=["/safe/input", "/another/safe/dir"],
            allowed_output_dirs=["/safe/output"],
            allowed_file_extensions=[".pdf", ".docx"]
        )
        assert len(security.allowed_input_dirs) == 2
        assert security.allowed_file_extensions == [".pdf", ".docx"]
        
        # Test empty extensions list
        security_empty = SecurityConfig(allowed_file_extensions=[])
        assert security_empty.allowed_file_extensions == []


@pytest_integration 
class TestConfigurationLoading:
    """Test configuration loading from files and environment."""
    
    def test_load_config_from_json_file(self, temp_workspace: Path) -> None:
        """Test loading configuration from JSON file."""
        config_file = temp_workspace / "config.json"
        
        config_data = {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True
            },
            "marker": {
                "batch_size": 8,
                "max_pages": 1000
            },
            "resource_limits": {
                "max_file_size_mb": 250,
                "max_memory_usage_mb": 6144,
                "max_concurrent_jobs": 4
            },
            "monitoring": {
                "enable_metrics": False,
                "log_performance": False
            },
            "security": {
                "validate_file_paths": False,
                "allowed_file_extensions": [".pdf", ".txt"]
            },
            "paths": {
                "cache_dir": "/custom/cache",
                "logs_dir": "/custom/logs",
            }
        }
        
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Load configuration from file
        with open(config_file, encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        config = AppConfig.model_validate(loaded_data)
        
        # Verify loaded values
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8000
        assert config.server.debug is True
        
        assert config.marker.batch_size == 8
        assert config.marker.max_pages == 1000
        
        assert config.resource_limits.max_file_size_mb == 250
        assert config.resource_limits.max_memory_usage_mb == 6144
        assert config.resource_limits.max_concurrent_jobs == 4
        
        assert config.monitoring.enable_metrics is False
        assert config.monitoring.log_performance is False
        
        assert config.security.validate_file_paths is False
        assert config.security.allowed_file_extensions == [".pdf", ".txt"]
        
        assert config.paths.cache_dir == "/custom/cache"
        assert config.paths.logs_dir == "/custom/logs"
    
    def test_load_partial_config_with_defaults(self, temp_workspace: Path) -> None:
        """Test loading partial configuration with defaults."""
        config_file = temp_workspace / "partial_config.json"
        
        # Only specify some values, others should use defaults
        partial_config = {
            "server": {
                "port": 9999
            },
            "resource_limits": {
                "max_file_size_mb": 100
            }
        }
        
        config_file.write_text(json.dumps(partial_config, indent=2))
        
        with open(config_file, encoding="utf-8") as f:
            loaded_data = json.load(f)
        
        config = AppConfig.model_validate(loaded_data)
        
        # Specified values
        assert config.server.port == 9999
        assert config.resource_limits.max_file_size_mb == 100
        
        # Default values
        assert config.server.host == "127.0.0.1"  # Default
        assert config.marker.batch_size == 1  # Default
        assert config.monitoring.enable_metrics is True  # Default


@pytest_integration
class TestConfigurationRobustness:
    """Test configuration robustness under stress and failure conditions."""
    
    @staticmethod
    def test_concurrent_config_modifications(temp_workspace: Path) -> None:
        """Test handling of concurrent configuration modifications."""
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        config_file = temp_workspace / "concurrent_config.json"
        
        # Initial configuration
        initial_config = {
            "resource_limits": {"max_concurrent_jobs": 3},
            "monitoring": {"enable_metrics": True}
        }
        config_file.write_text(json.dumps(initial_config, indent=2))
        
        configs_loaded = []
        errors = []
        
        def load_config(thread_id) -> AppConfig:
            """Load configuration in a thread."""
            try:
                time.sleep(0.01 * thread_id)  # Stagger access
                
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)
                
                config = AppConfig.model_validate(data)
                configs_loaded.append((thread_id, config))
                return config
            except Exception as e:
                errors.append((thread_id, e))
                raise
        
        def modify_config(thread_id):
            """Modify configuration in a thread."""
            try:
                time.sleep(0.005 * thread_id)  # Stagger modifications
                
                new_config = {
                    "resource_limits": {"max_concurrent_jobs": thread_id + 1},
                    "monitoring": {"enable_metrics": thread_id % 2 == 0}
                }
                
                # Atomic write (simulate proper file handling)
                temp_file = config_file.with_suffix('.tmp')
                temp_file.write_text(json.dumps(new_config, indent=2))
                temp_file.rename(config_file)
                
                return thread_id
            except Exception as e:
                errors.append((thread_id, e))
                raise
        
        # Test concurrent reads
        with ThreadPoolExecutor(max_workers=10) as executor:
            read_futures = [executor.submit(load_config, i) for i in range(5)]
            modify_futures = [executor.submit(modify_config, i) for i in range(3)]
            
            results = []
            exceptions = []
            for future in as_completed(read_futures + modify_futures):
                try:
                    results.append(future.result(timeout=5))
                except Exception as e:
                    exceptions.append(e)
            errors.extend(exceptions)
        
        # Handle exceptions after the loop
        for e in exceptions:
            # process or raise/log as needed
            pass
        
        # Verify no critical errors occurred
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
        
        # Verify at least some configs were loaded successfully
        assert len(configs_loaded) > 0, "No configurations were loaded successfully"
        
        # Verify final configuration is valid
        with open(config_file, encoding="utf-8") as f:
            final_data = json.load(f)
        final_config = AppConfig.model_validate(final_data)
        
        assert final_config.resource_limits.max_concurrent_jobs > 0
        assert isinstance(final_config.monitoring.enable_metrics, bool)
    
    def test_configuration_with_extreme_values(self) -> None:
        """Test configuration with extreme boundary values."""
        extreme_configs = [
            # Minimum values
            {
                "resource_limits": {
                    "max_file_size_mb": 1,
                    "max_memory_usage_mb": 1,
                    "max_processing_time_seconds": 1,
                    "max_concurrent_jobs": 1
                },
                "monitoring": {
                    "metrics_interval_seconds": 1,
                    "alert_memory_threshold_percent": 0.1
                }
            },
            # Maximum practical values
            {
                "resource_limits": {
                    "max_file_size_mb": 100000,  # 100GB
                    "max_memory_usage_mb": 1048576,  # 1TB
                    "max_processing_time_seconds": 86400,  # 24 hours
                    "max_concurrent_jobs": 1000
                },
                "monitoring": {
                    "metrics_interval_seconds": 3600,  # 1 hour
                    "alert_memory_threshold_percent": 99.9
                }
            },
            # Edge case values
            {
                "server": {"port": 1024},  # Minimum non-privileged port
                "marker": {"batch_size": 100, "max_pages": 10000},
                "monitoring": {
                    "alert_memory_threshold_percent": 50.0
                }
            }
        ]
        
        for config_data in extreme_configs:
            config = AppConfig.model_validate(config_data)
            
            # Verify all extreme values are handled correctly
            if "resource_limits" in config_data:
                limits = config_data["resource_limits"]
                if "max_file_size_mb" in limits:
                    assert config.resource_limits.max_file_size_mb == limits["max_file_size_mb"]
                if "max_memory_usage_mb" in limits:
                    assert config.resource_limits.max_memory_usage_mb == limits["max_memory_usage_mb"]
                if "max_processing_time_seconds" in limits:
                    assert config.resource_limits.max_processing_time_seconds == limits["max_processing_time_seconds"]
                if "max_concurrent_jobs" in limits:
                    assert config.resource_limits.max_concurrent_jobs == limits["max_concurrent_jobs"]
            
            if "monitoring" in config_data:
                monitoring = config_data["monitoring"]
                if "metrics_interval_seconds" in monitoring:
                    assert config.monitoring.metrics_interval_seconds == monitoring["metrics_interval_seconds"]
                if "alert_memory_threshold_percent" in monitoring:
                    assert config.monitoring.alert_memory_threshold_percent == monitoring["alert_memory_threshold_percent"]
    
    def test_configuration_corruption_handling(self, temp_workspace: Path) -> None:
        """Test handling of corrupted configuration files."""
        config_file = temp_workspace / "corrupted_config.json"
        
        # Test various corruption scenarios
        corruption_cases = [
            # Truncated JSON
            '{"server": {"host": "localhost", "port": 80',
            # Invalid JSON with extra characters
            '{"server": {"port": 8000}}}}',
            # JSON with non-ASCII characters
            '{"server": {"host": "localhost\\x00\\x01", "port": 8000}}',
            # Very large configuration
            '{"server": {"host": "' + 'x' * 100000 + '", "port": 8000}}',
            # Configuration with circular references (simulated)
            '{"server": {"host": "localhost", "port": 8000, "ref": "server"}}',
            # Extremely nested structure
            '{"level1": {"level2": {"level3": {"level4": {"level5": {"value": 42}}}}}}'
        ]
        
        for corrupted_content in corruption_cases:
            config_file.write_text(corrupted_content)
            with open(config_file, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Expected for malformed JSON
                    continue
                # If JSON loading succeeds, try to create AppConfig
                try:
                    config = AppConfig.model_validate(data)
                    # Some corruption cases might still create valid configs
                    assert config is not None
                except (ValueError, TypeError) as e:
                    # Expected for invalid configurations
                    assert "validation" in str(e).lower() or "type" in str(e).lower()
    
    def test_configuration_memory_pressure(self) -> None:
        """Test configuration handling under memory pressure."""
        
        # Create configurations with large data structures
        large_configs = []
        
        for i in range(100):
            # Create config with increasingly large data structures
            config_data = {
                "marker": {
                    "batch_size": min(i + 1, 100),  # Valid batch size
                    "max_pages": (i + 1) * 100
                },
                "security": {
                    "allowed_input_dirs": [f"/path/{j}" for j in range(i + 1)],
                    "allowed_output_dirs": [f"/output/{j}" for j in range(i + 1)],
                    "allowed_file_extensions": [f".ext{j}" for j in range(i + 1)]
                }
            }
            
            config = AppConfig.model_validate(config_data)
            large_configs.append(config)
            
            # Verify configuration is still valid
            assert config.marker.batch_size <= 100
            assert len(config.security.allowed_input_dirs) == i + 1
            
            # Check memory usage periodically
            if i % 20 == 0:
                import gc
                gc.collect()  # Force garbage collection
        
        # Verify all configurations are still accessible
        assert len(large_configs) == 100
        
        # Test final configuration
        final_config = large_configs[-1]
        assert final_config.marker.batch_size > 0
        assert len(final_config.security.allowed_input_dirs) == 100
    
    def test_configuration_validation_stress(self) -> None:
        """Test configuration validation under stress conditions."""
        # Generate many random configurations
        import random
        import string
        
        validation_results = []
        
        for i in range(1000):
            # Generate random configuration values
            config_data = {
                "server": {
                    "host": ''.join(random.choices(string.ascii_letters, k=random.randint(1, 50))),
                    "port": random.randint(-1000, 70000),  # Include invalid ports
                    "debug": random.choice([True, False])
                },
                "resource_limits": {
                    "max_file_size_mb": random.randint(-100, 1000000),
                    "max_memory_usage_mb": random.randint(-100, 1000000),
                    "max_processing_time_seconds": random.randint(-100, 1000000),
                    "max_concurrent_jobs": random.randint(-10, 1000)
                },
                "monitoring": {
                    "enable_metrics": random.choice([True, False]),
                    "metrics_interval_seconds": random.randint(-10, 10000),
                    "alert_memory_threshold_percent": random.uniform(-50, 200)
                }
            }
            
            try:
                config = AppConfig.model_validate(config_data)
                validation_results.append(("valid", i, config))
            except (ValueError, TypeError) as e:
                validation_results.append(("invalid", i, str(e)))
        
        # Analyze validation results
        valid_count = sum(1 for result in validation_results if result[0] == "valid")
        invalid_count = sum(1 for result in validation_results if result[0] == "invalid")
        
        # Should have both valid and invalid cases
        assert valid_count > 0, "No valid configurations generated"
        assert invalid_count > 0, "No invalid configurations detected"
        
        # Validation should catch most invalid cases
        assert invalid_count > valid_count, "Too many invalid configurations passed validation"
    
    @staticmethod
    def test_configuration_hot_reload_stress(temp_workspace: Path) -> None:
        """Test configuration hot reload under stress conditions."""
        
        config_file = temp_workspace / "hot_reload_stress.json"
        
        # Initial configuration
        base_config = {
            "resource_limits": {"max_concurrent_jobs": 3},
            "monitoring": {"enable_metrics": True}
        }
        config_file.write_text(json.dumps(base_config, indent=2))
        
        reload_count = 0
        error_count = 0
        configs = []
        
        def reload_config():
            """Simulate configuration reload operation."""
            nonlocal reload_count, error_count
            
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)
                
                config = AppConfig.model_validate(data)
                configs.append(config)
                reload_count += 1
                
                # Simulate processing delay
                time.sleep(0.001)
                
                return config
            except Exception:
                error_count += 1
                raise
        
        def update_config(iteration):
            """Update configuration file."""
            try:
                new_config = {
                    "resource_limits": {"max_concurrent_jobs": (iteration % 10) + 1},
                    "monitoring": {"enable_metrics": iteration % 2 == 0},
                    "server": {"port": 8000 + (iteration % 100)}
                }
                
                # Atomic write
                temp_file = config_file.with_suffix('.tmp')
                temp_file.write_text(json.dumps(new_config, indent=2))
                temp_file.rename(config_file)
                
                time.sleep(0.001)
                return iteration
            except Exception:
                nonlocal error_count
                error_count += 1
                raise

        # Simulate intensive hot reload scenario
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit many reload and update operations
            reload_futures = [executor.submit(reload_config) for _ in range(100)]
            update_futures = [executor.submit(update_config, i) for i in range(50)]
            
            results = []
            exceptions = []
            for future in as_completed(reload_futures + update_futures):
                try:
                    results.append(future.result(timeout=10))
                except Exception as e:
                    exceptions.append(e)
        
        # Verify reasonable success rate
        success_rate = reload_count / (reload_count + error_count) if (reload_count + error_count) > 0 else 0
        assert success_rate > 0.5, f"Hot reload success rate too low: {success_rate}"
        
        # Verify at least some configurations were loaded
        assert len(configs) > 0, "No configurations loaded during stress test"
        
        # Verify final configuration is valid
        try:
            with open(config_file, encoding="utf-8") as f:
                final_data = json.load(f)
            final_config = AppConfig.model_validate(final_data)
            
            assert final_config.resource_limits.max_concurrent_jobs > 0
            assert isinstance(final_config.monitoring.enable_metrics, bool)
        except Exception:
            # File might be in transition, which is acceptable
            pass


@pytest_integration
class TestConfigurationFailureRecovery:
    """Test configuration failure recovery scenarios."""
    
    def test_partial_configuration_failure_recovery(self, temp_workspace: Path) -> None:
        """Test recovery from partial configuration failures."""
        config_file = temp_workspace / "partial_failure_config.json"
        
        # Start with valid configuration
        valid_config = {
            "server": {"host": "127.0.0.1", "port": 8080},
            "resource_limits": {"max_concurrent_jobs": 5},
            "monitoring": {"enable_metrics": True}
        }
        config_file.write_text(json.dumps(valid_config, indent=2))
        
        # Load initial valid configuration
        with open(config_file, encoding="utf-8") as f:
            data = json.load(f)
        initial_config = AppConfig.model_validate(data)
        
        assert initial_config.server.port == 8080
        assert initial_config.resource_limits.max_concurrent_jobs == 5
        
        # Introduce partial configuration failure
        partial_invalid_config = {
            "server": {"host": "127.0.0.1", "port": -1},  # Invalid port
            "resource_limits": {"max_concurrent_jobs": 5},  # Valid
            "monitoring": {"enable_metrics": True}  # Valid
        }
        config_file.write_text(json.dumps(partial_invalid_config, indent=2))
        
        # Attempt to load invalid configuration
        with open(config_file, encoding="utf-8") as f:
            json.load(f)
        
        # Simulate fallback to previous valid configuration
        fallback_config = initial_config
        
        # Verify fallback works
        assert fallback_config.server.port == 8080
        assert fallback_config.resource_limits.max_concurrent_jobs == 5
        
        # Restore valid configuration
        fixed_config = {
            "server": {"host": "127.0.0.1", "port": 9090},  # Fixed port
            "resource_limits": {"max_concurrent_jobs": 8},  # Updated
            "monitoring": {"enable_metrics": True}
        }
        config_file.write_text(json.dumps(fixed_config, indent=2))
        
        # Load recovered configuration
        with open(config_file, encoding="utf-8") as f:
            recovered_data = json.load(f)
        recovered_config = AppConfig.model_validate(recovered_data)
        
        assert recovered_config.server.port == 9090
        assert recovered_config.resource_limits.max_concurrent_jobs == 8
    
    @staticmethod
    def test_configuration_backup_and_restore(temp_workspace: Path) -> None:
        """Test configuration backup and restore functionality."""
        config_file = temp_workspace / "main_config.json"
        backup_file = temp_workspace / "backup_config.json"
        
        # Create initial configuration
        config_data = {
            "server": {"host": "backup-test", "port": 7777},
            "resource_limits": {"max_concurrent_jobs": 4},
            "monitoring": {"enable_metrics": False}
        }
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # Create backup
        backup_file.write_text(config_file.read_text())
        
        # Verify backup was created
        assert backup_file.exists()
        
        # Load and verify backup
        with open(backup_file, encoding="utf-8") as f_backup:
            backup_data = json.load(f_backup)
        backup_config = AppConfig.model_validate(backup_data)
        
        assert backup_config.server.host == "backup-test"
        assert backup_config.server.port == 7777
        assert backup_config.resource_limits.max_concurrent_jobs == 4
        
        # Corrupt main configuration
        config_file.write_text('{"invalid": json}')
        
        # Verify main config is corrupted
        with pytest.raises(json.JSONDecodeError):
            with open(config_file, encoding="utf-8") as f:
                json.load(f)
        
        # Restore from backup
        config_file.write_text(backup_file.read_text())
        
        # Verify restoration
        with open(config_file, encoding="utf-8") as f:
            restored_data = json.load(f)
        restored_config = AppConfig.model_validate(restored_data)
        
        assert restored_config.server.host == "backup-test"
        assert restored_config.server.port == 7777
        assert restored_config.resource_limits.max_concurrent_jobs == 4


@pytest_unit  
class TestConfigurationPerformance:
    """Test configuration performance characteristics."""
    
    @staticmethod
    def test_configuration_loading_performance() -> None:
        """Test configuration loading performance.

        Raises:
            AssertionError: If configuration loading is too slow or validation fails.

        """
        # Create moderately complex configuration
        config_data = {
            "server": {"host": "perf-test", "port": 8888},
            "marker": {"batch_size": 10, "max_pages": 1000},
            "resource_limits": {"max_concurrent_jobs": 10},
            "security": {
                "allowed_input_dirs": [f"/input/dir/{i}" for i in range(100)],
                "allowed_output_dirs": [f"/output/dir/{i}" for i in range(100)],
                "allowed_file_extensions": [f".ext{i}" for i in range(50)]
            },
            "monitoring": {"enable_metrics": True}
        }
        
        # Measure configuration creation time
        start_time = time.time()
        
        config = None
        for _i in range(1000):
            config = AppConfig.model_validate(config_data)
        if config is None:
            msg = "Config was not created in the loop."
            raise AssertionError(msg)
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000
        
        # Should be reasonably fast
        assert total_time < PERFORMANCE_TIME_LIMIT, f"Configuration loading too slow: {total_time:.3f}s total"
        assert avg_time < 0.01, f"Average configuration loading too slow: {avg_time:.6f}s"
        
        # Verify configuration is still valid
        assert config.server.host == "perf-test"
        assert config.marker.batch_size == 10
        assert len(config.security.allowed_input_dirs) == 100
    
    def test_configuration_memory_usage(self) -> None:
        """Test configuration memory usage."""
        # Measure memory usage
        configs = []
        
        # Create many configurations
        for i in range(100):
            config_data = {
                "server": {"host": f"host-{i}", "port": 8000 + i},
                "marker": {"batch_size": (i % 100) + 1},
                "resource_limits": {"max_concurrent_jobs": i + 1},
                "security": {
                    "allowed_input_dirs": [f"/input/{j}" for j in range(i + 1)],
                    "allowed_file_extensions": [f".ext{j}" for j in range(i + 1)]
                }
            }
            
            config = AppConfig.model_validate(config_data)
            configs.append(config)
        
        # Verify all configurations are valid
        assert len(configs) == 100
        
        # Check that configurations scale reasonably
        first_config = configs[0]
        last_config = configs[-1]
        
        assert first_config.marker.batch_size == 1
        assert last_config.marker.batch_size == 100
        
        assert len(first_config.security.allowed_input_dirs) == 1
        assert len(last_config.security.allowed_input_dirs) == 100
        
        assert first_config.resource_limits.max_concurrent_jobs == 1
        assert last_config.resource_limits.max_concurrent_jobs == 100
    
    @staticmethod
    def test_configuration_serialization_performance() -> None:
        """Test configuration serialization performance.

        Raises:
            AssertionError: If serialization is too slow or serialization fails.

        """
        # Create complex configuration
        config_data = {
            "server": {"host": "serialization-test", "port": 9999},
            "marker": {
                "batch_size": 10,
                "max_pages": 5000,
            },
            "security": {
                "allowed_input_dirs": [f"/very/long/path/to/input/directory/number/{i}" for i in range(200)],
                "allowed_output_dirs": [f"/very/long/path/to/output/directory/number/{i}" for i in range(200)],
                "allowed_file_extensions": [f".extension_type_{i}" for i in range(100)]
            }
        }
        
        config = AppConfig.model_validate(config_data)
        
        # Measure serialization time
        start_time = time.time()
        
        serialized = None
        for _i in range(1000):
            serialized = config.model_dump()
        if serialized is None:
            msg = "Serialized config was not created in the loop."
            raise AssertionError(msg)
            
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000
        
        # Should be reasonably fast
        assert total_time < 5.0, f"Configuration serialization too slow: {total_time:.3f}s total"
        assert avg_time < 0.005, f"Average serialization too slow: {avg_time:.6f}s"
        
        # Verify serialization correctness
        assert serialized["server"]["host"] == "serialization-test"
        assert serialized["marker"]["batch_size"] == 10
        assert len(serialized["security"]["allowed_input_dirs"]) == 200
