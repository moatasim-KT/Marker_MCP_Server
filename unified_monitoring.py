#!/usr/bin/env python3
"""
Unified Monitoring System for Marker
Consolidates all monitoring functionality into a single comprehensive system.
"""

import sqlite3
import json
import psutil
import time
import os
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, asdict
import pandas as pd
import threading
import logging
from contextlib import contextmanager

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    process_id: Optional[int]
    cpu_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    status: str

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    timestamp: datetime
    test_name: str
    success_rate: float
    processing_time: float
    output_quality_score: float
    error_count: int
    warning_count: int

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str    # 'low', 'medium', 'high', 'critical'
    enabled: bool = True

# =============================================================================
# Unified Monitoring System
# =============================================================================

class UnifiedMonitoringSystem:
    """Complete monitoring system for Marker processing"""
    
    def __init__(self, db_path: str = "monitoring/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.monitoring = False
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.metrics_history = []
        
        # Alert system
        self.alert_rules = []
        self.alerts_history = []
        
        # Setup database and logging
        self._setup_database()
        self._setup_logging()
        self._setup_default_alerts()
    
    def _setup_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Performance metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    process_id INTEGER,
                    cpu_percent REAL,
                    memory_mb REAL,
                    disk_io_read_mb REAL,
                    disk_io_write_mb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    open_files INTEGER,
                    status TEXT
                )
            ''')
            
            # Quality metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    test_name TEXT,
                    success_rate REAL,
                    processing_time REAL,
                    output_quality_score REAL,
                    error_count INTEGER,
                    warning_count INTEGER
                )
            ''')
            
            # Alerts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    rule_name TEXT,
                    severity TEXT,
                    message TEXT,
                    metric_value REAL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def _setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.db_path.parent / 'monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule("High CPU Usage", "cpu_percent", 80.0, "gt", "high"),
            AlertRule("High Memory Usage", "memory_mb", 2048.0, "gt", "medium"),
            AlertRule("Low Success Rate", "success_rate", 0.8, "lt", "critical"),
            AlertRule("High Error Count", "error_count", 5, "gt", "high"),
            AlertRule("Slow Processing", "processing_time", 300.0, "gt", "medium"),
            AlertRule("Poor Quality Score", "output_quality_score", 0.7, "lt", "high")
        ]
        self.alert_rules.extend(default_rules)
    
    # =============================================================================
    # Performance Monitoring
    # =============================================================================
    
    def find_marker_processes(self) -> List[psutil.Process]:
        """Find all running marker processes"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('marker' in str(arg).lower() for arg in proc.info['cmdline']):
                    processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        # Get system-wide metrics (non-blocking)
        cpu_percent = psutil.cpu_percent(interval=0)  # Non-blocking
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Try to find marker process
        marker_processes = self.find_marker_processes()
        process_id = marker_processes[0].pid if marker_processes else None
        try:
            open_files = len(marker_processes[0].open_files()) if marker_processes else 0
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            open_files = 0
        status = "running" if marker_processes else "idle"
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            process_id=process_id,
            cpu_percent=cpu_percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_io_read_mb=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
            network_sent_mb=network_io.bytes_sent / 1024 / 1024 if network_io else 0,
            network_recv_mb=network_io.bytes_recv / 1024 / 1024 if network_io else 0,
            open_files=open_files,
            status=status
        )
    
    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_metrics 
                (timestamp, process_id, cpu_percent, memory_mb, disk_io_read_mb, 
                 disk_io_write_mb, network_sent_mb, network_recv_mb, open_files, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.process_id,
                metrics.cpu_percent,
                metrics.memory_mb,
                metrics.disk_io_read_mb,
                metrics.disk_io_write_mb,
                metrics.network_sent_mb,
                metrics.network_recv_mb,
                metrics.open_files,
                metrics.status
            ))
            conn.commit()
    
    def record_quality_metrics(self, metrics: QualityMetrics):
        """Record quality metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_metrics 
                (timestamp, test_name, success_rate, processing_time, 
                 output_quality_score, error_count, warning_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.test_name,
                metrics.success_rate,
                metrics.processing_time,
                metrics.output_quality_score,
                metrics.error_count,
                metrics.warning_count
            ))
            conn.commit()
    
    def log_quality_metrics(self, test_name: str, metrics_dict: Dict[str, Any]):
        """Convenience method to log quality metrics from a dictionary"""
        quality_metrics = QualityMetrics(
            timestamp=datetime.now(),
            test_name=test_name,
            success_rate=metrics_dict.get('success_rate', 1.0),
            processing_time=metrics_dict.get('processing_time', 0.0),
            output_quality_score=metrics_dict.get('output_quality_score', 0.9),
            error_count=metrics_dict.get('error_count', 0),
            warning_count=metrics_dict.get('warning_count', 0)
        )
        self.record_quality_metrics(quality_metrics)
    
    # =============================================================================
    # Alert System
    # =============================================================================
    
    def check_alerts(self, metrics: Dict[str, float]):
        """Check metrics against alert rules"""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            if rule.metric not in metrics:
                continue
                
            value = metrics[rule.metric]
            triggered = False
            
            if rule.comparison == "gt" and value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and value == rule.threshold:
                triggered = True
            
            if triggered:
                self._trigger_alert(rule, value)
    
    def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger an alert"""
        message = f"{rule.name}: {rule.metric} = {value:.2f} (threshold: {rule.threshold})"
        
        # Record alert
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alerts (timestamp, rule_name, severity, message, metric_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                rule.name,
                rule.severity,
                message,
                value
            ))
            conn.commit()
        
        # Log alert
        severity_emoji = {
            "low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"
        }
        emoji = severity_emoji.get(rule.severity, "⚠️")
        self.logger.warning(f"{emoji} ALERT [{rule.severity.upper()}]: {message}")
    
    # =============================================================================
    # Context Manager for Monitoring
    # =============================================================================
    
    @contextmanager
    def monitor_performance(self, test_name: str = "unknown"):
        """Context manager for automatic performance monitoring"""
        start_time = time.time()
        start_metrics = self.collect_system_metrics()
        
        self.logger.info(f"🚀 Starting monitoring for: {test_name}")
        
        try:
            yield self
        except Exception as e:
            self.logger.error(f"❌ Error during {test_name}: {e}")
            raise
        finally:
            end_time = time.time()
            end_metrics = self.collect_system_metrics()
            
            # Record performance
            self.record_performance_metrics(end_metrics)
            
            # Record quality (basic metrics)
            processing_time = end_time - start_time
            quality_metrics = QualityMetrics(
                timestamp=datetime.now(),
                test_name=test_name,
                success_rate=1.0,  # Assume success if no exception
                processing_time=processing_time,
                output_quality_score=0.9,  # Default good score
                error_count=0,
                warning_count=0
            )
            self.record_quality_metrics(quality_metrics)
            
            # Check alerts
            self.check_alerts({
                "cpu_percent": end_metrics.cpu_percent,
                "memory_mb": end_metrics.memory_mb,
                "processing_time": processing_time,
                "success_rate": 1.0,
                "output_quality_score": 0.9,
                "error_count": 0
            })
            
            self.logger.info(f"✅ Completed monitoring for: {test_name} ({processing_time:.2f}s)")
    
    # =============================================================================
    # Real-time Monitoring
    # =============================================================================
    
    def start_realtime_monitoring(self, interval: int = 5):
        """Start real-time monitoring in background"""
        if self.monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self.monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info(f"🔍 Started real-time monitoring (interval: {interval}s)")
    
    def stop_realtime_monitoring(self):
        """Stop real-time monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("🛑 Stopped real-time monitoring")
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while not self.stop_event.wait(interval):
            try:
                metrics = self.collect_system_metrics()
                self.record_performance_metrics(metrics)
                
                # Check for alerts
                self.check_alerts({
                    "cpu_percent": metrics.cpu_percent,
                    "memory_mb": metrics.memory_mb
                })
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    # =============================================================================
    # Dashboard and Reporting
    # =============================================================================
    
    def get_performance_data(self, days: int = 7) -> pd.DataFrame:
        """Get performance metrics from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM performance_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=[cutoff_date.isoformat()])
    
    def get_quality_data(self, days: int = 7) -> pd.DataFrame:
        """Get quality metrics from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM quality_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=[cutoff_date.isoformat()])
    
    def get_alerts_data(self, days: int = 7) -> pd.DataFrame:
        """Get alerts from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM alerts 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=[cutoff_date.isoformat()])
    
    def generate_performance_plot(self, days: int = 7, save_path: Optional[str] = None):
        """Generate performance visualization"""
        df = self.get_performance_data(days)
        
        if df.empty:
            self.logger.warning("No performance data available for plotting")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Marker Performance Metrics - Last {days} Days', fontsize=16)
        
        # CPU Usage
        axes[0,0].plot(df['timestamp'], df['cpu_percent'], 'b-', alpha=0.7)
        axes[0,0].set_title('CPU Usage (%)')
        axes[0,0].set_ylabel('CPU %')
        axes[0,0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0,1].plot(df['timestamp'], df['memory_mb'], 'r-', alpha=0.7)
        axes[0,1].set_title('Memory Usage (MB)')
        axes[0,1].set_ylabel('Memory MB')
        axes[0,1].grid(True, alpha=0.3)
        
        # Disk I/O
        axes[1,0].plot(df['timestamp'], df['disk_io_read_mb'], 'g-', alpha=0.7, label='Read')
        axes[1,0].plot(df['timestamp'], df['disk_io_write_mb'], 'orange', alpha=0.7, label='Write')
        axes[1,0].set_title('Disk I/O (MB)')
        axes[1,0].set_ylabel('Disk I/O MB')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Network I/O
        axes[1,1].plot(df['timestamp'], df['network_sent_mb'], 'purple', alpha=0.7, label='Sent')
        axes[1,1].plot(df['timestamp'], df['network_recv_mb'], 'brown', alpha=0.7, label='Received')
        axes[1,1].set_title('Network I/O (MB)')
        axes[1,1].set_ylabel('Network I/O MB')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plot saved to: {save_path}")
        else:
            plt.show()
    
    def print_status_summary(self):
        """Print current monitoring status"""
        print("\n" + "="*60)
        print("🔍 UNIFIED MONITORING SYSTEM STATUS")
        print("="*60)
        
        # System status
        marker_processes = self.find_marker_processes()
        print(f"📊 System Status:")
        print(f"   • Marker Processes: {len(marker_processes)}")
        print(f"   • Real-time Monitoring: {'🟢 Active' if self.monitoring else '🔴 Inactive'}")
        
        # Recent metrics
        try:
            recent_perf = self.get_performance_data(days=1)
            recent_quality = self.get_quality_data(days=1)
            recent_alerts = self.get_alerts_data(days=1)
            
            print(f"\n📈 Recent Data (24h):")
            print(f"   • Performance Records: {len(recent_perf)}")
            print(f"   • Quality Records: {len(recent_quality)}")
            print(f"   • Active Alerts: {len(recent_alerts[recent_alerts['resolved'] == False])}")
            
            if not recent_perf.empty:
                latest = recent_perf.iloc[0]
                print(f"\n🔧 Current Metrics:")
                print(f"   • CPU Usage: {latest['cpu_percent']:.1f}%")
                print(f"   • Memory Usage: {latest['memory_mb']:.1f} MB")
                print(f"   • Status: {latest['status']}")
                
        except Exception as e:
            print(f"   • Error reading metrics: {e}")
        
        print("="*60)

# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI interface for unified monitoring"""
    parser = argparse.ArgumentParser(description="Unified Marker Monitoring System")
    parser.add_argument("--start", action="store_true", help="Start real-time monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop real-time monitoring")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    parser.add_argument("--days", type=int, default=7, help="Number of days for data analysis")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize monitoring system
    monitor = UnifiedMonitoringSystem()
    
    if args.start:
        monitor.start_realtime_monitoring(args.interval)
        print(f"✅ Started real-time monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop...")
        try:
            while True: # Keep main thread alive
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_realtime_monitoring() # Ensure stop is called
            print("\\n🛑 Monitoring stopped by user") # Add user feedback
    
    elif args.stop:
        monitor.stop_realtime_monitoring()
        print("🛑 Monitoring stopped") # Ensure message is printed
    
    elif args.plot:
        print(f"📊 Generating performance plot for the last {args.days} days...")
        monitor.generate_performance_plot(args.days)
    
    else: # Default to status if no other action specified or if args.status is true
        monitor.print_status_summary()

if __name__ == "__main__":
    main()
