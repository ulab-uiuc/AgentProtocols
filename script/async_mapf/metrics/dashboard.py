"""
Real-time dashboard for MAPF metrics monitoring.

Provides live visualization of MAPF execution metrics.
"""

import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class DashboardMetric:
    """Single dashboard metric."""
    name: str
    value: Any
    timestamp: float
    category: str = "general"
    unit: str = ""
    description: str = ""


class RealtimeDashboard:
    """
    Real-time dashboard for monitoring MAPF execution.
    
    Displays live metrics and provides alerts for anomalies.
    """
    
    def __init__(self, update_interval: float = 1.0, max_history: int = 1000):
        """
        Initialize real-time dashboard.
        
        Args:
            update_interval: Update frequency in seconds
            max_history: Maximum number of metric entries to keep
        """
        self.update_interval = update_interval
        self.max_history = max_history
        
        # Metric storage
        self.current_metrics: Dict[str, DashboardMetric] = {}
        self.metric_history: Dict[str, List[DashboardMetric]] = {}
        
        # Dashboard state
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
        # Callbacks and alerts
        self.metric_callbacks: Dict[str, List[Callable]] = {}
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Data sources
        self.data_sources: List[Callable] = []
    
    def start(self) -> None:
        """Start the real-time dashboard."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        print("Real-time dashboard started")
    
    def stop(self) -> None:
        """Stop the real-time dashboard."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        print("Real-time dashboard stopped")
    
    def add_data_source(self, source_func: Callable[[], Dict[str, Any]]) -> None:
        """
        Add a data source function.
        
        Args:
            source_func: Function that returns metric data
        """
        self.data_sources.append(source_func)
    
    def update_metric(self, name: str, value: Any, category: str = "general", 
                     unit: str = "", description: str = "") -> None:
        """
        Update a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            category: Metric category
            unit: Value unit
            description: Metric description
        """
        metric = DashboardMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            category=category,
            unit=unit,
            description=description
        )
        
        self.current_metrics[name] = metric
        
        # Add to history
        if name not in self.metric_history:
            self.metric_history[name] = []
        
        self.metric_history[name].append(metric)
        
        # Trim history if too long
        if len(self.metric_history[name]) > self.max_history:
            self.metric_history[name] = self.metric_history[name][-self.max_history:]
        
        # Check alerts
        self._check_alerts(metric)
        
        # Trigger callbacks
        self._trigger_callbacks(name, metric)
    
    def add_metric_callback(self, metric_name: str, callback: Callable[[DashboardMetric], None]) -> None:
        """
        Add callback for metric updates.
        
        Args:
            metric_name: Name of metric to monitor
            callback: Callback function
        """
        if metric_name not in self.metric_callbacks:
            self.metric_callbacks[metric_name] = []
        
        self.metric_callbacks[metric_name].append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold_type: str, 
                           threshold_value: float, message: str = "") -> None:
        """
        Set alert threshold for a metric.
        
        Args:
            metric_name: Name of metric
            threshold_type: Type of threshold ("min", "max", "change_rate")
            threshold_value: Threshold value
            message: Alert message template
        """
        if metric_name not in self.alert_thresholds:
            self.alert_thresholds[metric_name] = {}
        
        self.alert_thresholds[metric_name][threshold_type] = {
            "value": threshold_value,
            "message": message
        }
    
    def get_current_metrics(self) -> Dict[str, DashboardMetric]:
        """Get all current metrics."""
        return self.current_metrics.copy()
    
    def get_metric_history(self, metric_name: str, 
                          time_window: Optional[float] = None) -> List[DashboardMetric]:
        """
        Get metric history.
        
        Args:
            metric_name: Name of metric
            time_window: Time window in seconds (None for all history)
            
        Returns:
            List of metric entries
        """
        if metric_name not in self.metric_history:
            return []
        
        history = self.metric_history[metric_name]
        
        if time_window is None:
            return history.copy()
        
        # Filter by time window
        cutoff_time = time.time() - time_window
        return [m for m in history if m.timestamp >= cutoff_time]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return self.active_alerts.copy()
    
    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self.active_alerts.clear()
    
    def export_current_state(self) -> Dict[str, Any]:
        """Export current dashboard state."""
        return {
            "timestamp": time.time(),
            "metrics": {
                name: {
                    "name": metric.name,
                    "value": metric.value,
                    "timestamp": metric.timestamp,
                    "category": metric.category,
                    "unit": metric.unit,
                    "description": metric.description
                }
                for name, metric in self.current_metrics.items()
            },
            "alerts": self.active_alerts.copy()
        }
    
    def _update_loop(self) -> None:
        """Main update loop running in background thread."""
        while self.is_running:
            try:
                # Collect data from all sources
                for source_func in self.data_sources:
                    try:
                        source_data = source_func()
                        self._process_source_data(source_data)
                    except Exception as e:
                        print(f"Dashboard: Error collecting data from source: {e}")
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Dashboard: Error in update loop: {e}")
                time.sleep(self.update_interval)
    
    def _process_source_data(self, data: Dict[str, Any]) -> None:
        """Process data from a source."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested data structure
                category = key
                for sub_key, sub_value in value.items():
                    metric_name = f"{category}.{sub_key}"
                    self.update_metric(metric_name, sub_value, category=category)
            else:
                # Direct value
                self.update_metric(key, value)
    
    def _check_alerts(self, metric: DashboardMetric) -> None:
        """Check if metric triggers any alerts."""
        thresholds = self.alert_thresholds.get(metric.name, {})
        
        for threshold_type, threshold_config in thresholds.items():
            threshold_value = threshold_config["value"]
            message_template = threshold_config.get("message", "")
            
            alert_triggered = False
            alert_message = ""
            
            if threshold_type == "max" and isinstance(metric.value, (int, float)):
                if metric.value > threshold_value:
                    alert_triggered = True
                    alert_message = f"{metric.name} exceeded maximum threshold: {metric.value} > {threshold_value}"
            
            elif threshold_type == "min" and isinstance(metric.value, (int, float)):
                if metric.value < threshold_value:
                    alert_triggered = True
                    alert_message = f"{metric.name} below minimum threshold: {metric.value} < {threshold_value}"
            
            elif threshold_type == "change_rate" and isinstance(metric.value, (int, float)):
                # Check rate of change
                history = self.metric_history.get(metric.name, [])
                if len(history) >= 2:
                    prev_metric = history[-2]
                    time_diff = metric.timestamp - prev_metric.timestamp
                    if time_diff > 0:
                        change_rate = abs(metric.value - prev_metric.value) / time_diff
                        if change_rate > threshold_value:
                            alert_triggered = True
                            alert_message = f"{metric.name} change rate exceeded threshold: {change_rate:.2f}/s > {threshold_value}/s"
            
            if alert_triggered:
                alert = {
                    "metric_name": metric.name,
                    "threshold_type": threshold_type,
                    "threshold_value": threshold_value,
                    "current_value": metric.value,
                    "message": message_template or alert_message,
                    "timestamp": metric.timestamp
                }
                
                # Check if alert is already active
                alert_exists = any(
                    a["metric_name"] == metric.name and a["threshold_type"] == threshold_type
                    for a in self.active_alerts
                )
                
                if not alert_exists:
                    self.active_alerts.append(alert)
                    print(f"ALERT: {alert['message']}")
    
    def _trigger_callbacks(self, metric_name: str, metric: DashboardMetric) -> None:
        """Trigger callbacks for metric updates."""
        callbacks = self.metric_callbacks.get(metric_name, [])
        
        for callback in callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"Dashboard: Error in metric callback: {e}")
    
    def print_summary(self) -> None:
        """Print current dashboard summary to console."""
        print("\n" + "="*60)
        print("MAPF Real-time Dashboard Summary")
        print("="*60)
        
        # Group metrics by category
        categories = {}
        for metric in self.current_metrics.values():
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append(metric)
        
        # Print metrics by category
        for category, metrics in categories.items():
            print(f"\n{category.upper()}:")
            for metric in sorted(metrics, key=lambda m: m.name):
                unit_str = f" {metric.unit}" if metric.unit else ""
                print(f"  {metric.name}: {metric.value}{unit_str}")
        
        # Print active alerts
        if self.active_alerts:
            print(f"\nALERTS ({len(self.active_alerts)} active):")
            for alert in self.active_alerts[-5:]:  # Show last 5 alerts
                print(f"  {alert['message']}")
        
        print("="*60)
    
    def generate_web_dashboard(self, output_file: str = "dashboard.html") -> str:
        """
        Generate a simple HTML dashboard.
        
        Args:
            output_file: Output HTML file path
            
        Returns:
            Path to generated HTML file
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MAPF Real-time Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-category { margin-bottom: 20px; }
        .metric-category h2 { color: #333; border-bottom: 2px solid #007bff; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-width: 200px; }
        .metric-name { font-weight: bold; }
        .metric-value { font-size: 1.2em; color: #007bff; }
        .alert { background-color: #ffe6e6; border-left: 5px solid #ff0000; padding: 10px; margin: 5px 0; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }
        setInterval(refreshPage, 5000); // Refresh every 5 seconds
    </script>
</head>
<body>
    <h1>MAPF Real-time Dashboard</h1>
    <p class="timestamp">Last updated: {timestamp}</p>
    
    {alerts_html}
    
    {metrics_html}
</body>
</html>
        """
        
        # Generate alerts HTML
        alerts_html = ""
        if self.active_alerts:
            alerts_html = "<h2>Active Alerts</h2>"
            for alert in self.active_alerts:
                alerts_html += f'<div class="alert">{alert["message"]}</div>'
        
        # Generate metrics HTML
        categories = {}
        for metric in self.current_metrics.values():
            if metric.category not in categories:
                categories[metric.category] = []
            categories[metric.category].append(metric)
        
        metrics_html = ""
        for category, metrics in categories.items():
            metrics_html += f'<div class="metric-category">'
            metrics_html += f'<h2>{category.upper()}</h2>'
            
            for metric in sorted(metrics, key=lambda m: m.name):
                unit_str = f" {metric.unit}" if metric.unit else ""
                metrics_html += f'''
                <div class="metric">
                    <div class="metric-name">{metric.name}</div>
                    <div class="metric-value">{metric.value}{unit_str}</div>
                    <div class="timestamp">{time.strftime("%H:%M:%S", time.localtime(metric.timestamp))}</div>
                </div>
                '''
            
            metrics_html += '</div>'
        
        # Fill template
        html_content = html_template.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            alerts_html=alerts_html,
            metrics_html=metrics_html
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file


class MAPFDashboardConnector:
    """
    Connector to integrate MAPF components with dashboard.
    
    Automatically collects metrics from MAPF network and agents.
    """
    
    def __init__(self, dashboard: RealtimeDashboard):
        """
        Initialize connector.
        
        Args:
            dashboard: Dashboard instance to update
        """
        self.dashboard = dashboard
        self.network = None
        self.agents = []
    
    def connect_network(self, network) -> None:
        """Connect to MAPF network coordinator."""
        self.network = network
        self.dashboard.add_data_source(self._collect_network_metrics)
    
    def connect_agents(self, agents: List) -> None:
        """Connect to MAPF agents."""
        self.agents = agents
        self.dashboard.add_data_source(self._collect_agent_metrics)
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect metrics from network coordinator."""
        if not self.network:
            return {}
        
        try:
            if hasattr(self.network, 'get_performance_metrics'):
                metrics = self.network.get_performance_metrics()
                return {"network": metrics}
        except Exception as e:
            return {"network_error": str(e)}
        
        return {}
    
    def _collect_agent_metrics(self) -> Dict[str, Any]:
        """Collect metrics from agents."""
        if not self.agents:
            return {}
        
        try:
            agent_data = {}
            
            for agent in self.agents:
                if hasattr(agent, 'get_connection_status'):
                    status = agent.get_connection_status()
                    agent_data[f"agent_{agent.aid}"] = status
                
                # Basic agent info
                agent_data[f"agent_{agent.aid}_goal_reached"] = agent.is_at_goal()
                agent_data[f"agent_{agent.aid}_active"] = agent.is_active
            
            return agent_data
            
        except Exception as e:
            return {"agent_error": str(e)}
        
        return {} 