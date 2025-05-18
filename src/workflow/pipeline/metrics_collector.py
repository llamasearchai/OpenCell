import datetime
import json
import logging
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""

    COUNTER = auto()  # Incrementing counters (e.g., number of processed items)
    GAUGE = auto()  # Point-in-time values (e.g., memory usage)
    HISTOGRAM = auto()  # Distribution of values (e.g., processing times)
    TIMING = auto()  # Time duration measurements
    EVENT = auto()  # Timestamped events
    LABELED = auto()  # Values with multiple labels


@dataclass
class MetricsCollector:
    """Enhanced metrics collector with more capabilities"""

    metrics_dir: str = "./metrics"
    flush_interval: int = 60  # Flush metrics to disk every 60 seconds
    enabled: bool = True
    max_items_per_metric: int = 1000  # Maximum items to keep in memory per metric
    retention_days: int = 7  # Number of days to keep metrics files

    # Storage for metrics
    counters: Dict[str, int] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, List[float]] = field(default_factory=list)
    timings: Dict[str, List[float]] = field(default_factory=list)
    events: Dict[str, List[Dict[str, Any]]] = field(default_factory=list)
    labeled_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Track metrics metadata
    metric_types: Dict[str, MetricType] = field(default_factory=dict)
    metric_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # For visualization
    visualization_port: Optional[int] = None
    visualization_server: Any = None

    # Add new metrics storage
    performance_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    resource_history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize metrics collector"""
        self.metrics_dir = Path(self.metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.last_flush_time = time.time()

        # Create subdirectories for different metric types
        for metric_type in MetricType:
            (self.metrics_dir / metric_type.name.lower()).mkdir(exist_ok=True)

        # Start flush thread if enabled
        if self.enabled and self.flush_interval > 0:
            self._start_flush_thread()

        # Start visualization server if port specified
        if self.visualization_port:
            self._start_visualization_server()

    def _start_flush_thread(self):
        """Start a background thread to periodically flush metrics"""

        def flush_metrics():
            while True:
                time.sleep(self.flush_interval)
                try:
                    self.flush()
                except Exception as e:
                    logger.error(f"Error flushing metrics: {str(e)}")

        flush_thread = threading.Thread(target=flush_metrics, daemon=True)
        flush_thread.start()
        logger.info(f"Started metrics flush thread with interval {self.flush_interval}s")

    def _start_visualization_server(self):
        """Start a web server for metrics visualization"""
        try:
            from flask import Flask, jsonify, render_template_string

            app = Flask("MetricsVisualization")

            @app.route("/")
            def index():
                """Render metrics dashboard"""
                return render_template_string(
                    """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Metrics Dashboard</title>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .container { display: flex; flex-wrap: wrap; }
                        .metric-card { 
                            background: #f9f9f9; 
                            border-radius: 5px; 
                            padding: 15px; 
                            margin: 10px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            min-width: 300px;
                            flex: 1;
                        }
                        h1, h2 { color: #333; }
                    </style>
                </head>
                <body>
                    <h1>Metrics Dashboard</h1>
                    <div>
                        <button onclick="refreshData()">Refresh</button>
                        <span id="last-update"></span>
                    </div>
                    <div class="container" id="counters-container">
                        <h2>Counters</h2>
                    </div>
                    <div class="container" id="gauges-container">
                        <h2>Gauges</h2>
                    </div>
                    <div class="container" id="histograms-container">
                        <h2>Histograms</h2>
                    </div>
                    <div class="container" id="timings-container">
                        <h2>Timings</h2>
                    </div>
                    
                    <script>
                        const charts = {};
                        
                        function createMetricCard(container, title) {
                            const card = document.createElement('div');
                            card.className = 'metric-card';
                            card.innerHTML = `<h3>${title}</h3><canvas id="chart-${title}"></canvas>`;
                            document.getElementById(container).appendChild(card);
                            return `chart-${title}`;
                        }
                        
                        function refreshData() {
                            fetch('/api/metrics')
                                .then(response => response.json())
                                .then(data => {
                                    updateCounters(data.counters);
                                    updateGauges(data.gauges);
                                    updateHistograms(data.histograms);
                                    updateTimings(data.timings);
                                    
                                    document.getElementById('last-update').textContent = 
                                        `Last updated: ${new Date().toLocaleTimeString()}`;
                                });
                        }
                        
                        function updateCounters(counters) {
                            const container = document.getElementById('counters-container');
                            for (const [name, value] of Object.entries(counters)) {
                                if (!charts[name]) {
                                    const canvasId = createMetricCard('counters-container', name);
                                    const ctx = document.getElementById(canvasId).getContext('2d');
                                    charts[name] = new Chart(ctx, {
                                        type: 'bar',
                                        data: {
                                            labels: [name],
                                            datasets: [{
                                                label: 'Value',
                                                data: [value],
                                                backgroundColor: 'rgba(54, 162, 235, 0.5)'
                                            }]
                                        },
                                        options: {
                                            scales: {
                                                y: {
                                                    beginAtZero: true
                                                }
                                            }
                                        }
                                    });
                                } else {
                                    charts[name].data.datasets[0].data = [value];
                                    charts[name].update();
                                }
                            }
                        }
                        
                        function updateGauges(gauges) {
                            const container = document.getElementById('gauges-container');
                            for (const [name, value] of Object.entries(gauges)) {
                                if (!charts[name]) {
                                    const canvasId = createMetricCard('gauges-container', name);
                                    const ctx = document.getElementById(canvasId).getContext('2d');
                                    charts[name] = new Chart(ctx, {
                                        type: 'doughnut',
                                        data: {
                                            labels: ['Value'],
                                            datasets: [{
                                                data: [value, 100 - value],
                                                backgroundColor: [
                                                    'rgba(75, 192, 192, 0.5)',
                                                    'rgba(200, 200, 200, 0.5)'
                                                ]
                                            }]
                                        },
                                        options: {
                                            circumference: 180,
                                            rotation: -90,
                                            plugins: {
                                                legend: {
                                                    display: false
                                                }
                                            }
                                        }
                                    });
                                } else {
                                    charts[name].data.datasets[0].data = [value, 100 - value];
                                    charts[name].update();
                                }
                            }
                        }
                        
                        function updateHistograms(histograms) {
                            const container = document.getElementById('histograms-container');
                            for (const [name, data] of Object.entries(histograms)) {
                                if (!charts[name]) {
                                    const canvasId = createMetricCard('histograms-container', name);
                                    const ctx = document.getElementById(canvasId).getContext('2d');
                                    charts[name] = new Chart(ctx, {
                                        type: 'line',
                                        data: {
                                            labels: Array.from({length: data.recent.length}, (_, i) => i + 1),
                                            datasets: [{
                                                label: 'Recent Values',
                                                data: data.recent,
                                                borderColor: 'rgba(153, 102, 255, 1)',
                                                fill: false
                                            }]
                                        }
                                    });
                                } else {
                                    charts[name].data.labels = Array.from({length: data.recent.length}, (_, i) => i + 1);
                                    charts[name].data.datasets[0].data = data.recent;
                                    charts[name].update();
                                }
                            }
                        }
                        
                        function updateTimings(timings) {
                            const container = document.getElementById('timings-container');
                            for (const [name, data] of Object.entries(timings)) {
                                if (!charts[name]) {
                                    const canvasId = createMetricCard('timings-container', name);
                                    const ctx = document.getElementById(canvasId).getContext('2d');
                                    charts[name] = new Chart(ctx, {
                                        type: 'bar',
                                        data: {
                                            labels: ['Min', 'Avg', 'P90', 'Max'],
                                            datasets: [{
                                                label: 'Timing (ms)',
                                                data: [data.min, data.avg, data.p90, data.max],
                                                backgroundColor: [
                                                    'rgba(75, 192, 192, 0.5)',
                                                    'rgba(54, 162, 235, 0.5)',
                                                    'rgba(255, 206, 86, 0.5)',
                                                    'rgba(255, 99, 132, 0.5)'
                                                ]
                                            }]
                                        }
                                    });
                                } else {
                                    charts[name].data.datasets[0].data = [data.min, data.avg, data.p90, data.max];
                                    charts[name].update();
                                }
                            }
                        }
                        
                        // Initial load
                        refreshData();
                        
                        // Auto refresh every 5 seconds
                        setInterval(refreshData, 5000);
                    </script>
                </body>
                </html>
                """
                )

            @app.route("/api/metrics")
            def get_metrics():
                """API endpoint to get metrics data"""
                return jsonify(self.get_all_metrics())

            def run_server():
                """Run Flask server in a separate thread"""
                app.run(host="0.0.0.0", port=self.visualization_port, debug=False, use_reloader=False)

            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            self.visualization_server = app
            logger.info(f"Started metrics visualization server on port {self.visualization_port}")

        except ImportError:
            logger.warning("Flask not installed. Metrics visualization server not started.")

    def counter(self, name: str, value: int = 1, metadata: Dict[str, Any] = None):
        """
        Increment a counter metric

        Args:
            name: Name of the counter
            value: Value to increment by
            metadata: Optional metadata for this metric
        """
        if not self.enabled:
            return

        if name not in self.counters:
            self.counters[name] = 0
            self.metric_types[name] = MetricType.COUNTER
            if metadata:
                self.metric_metadata[name] = metadata

        self.counters[name] += value

    def gauge(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Set a gauge metric

        Args:
            name: Name of the gauge
            value: Value to set
            metadata: Optional metadata for this metric
        """
        if not self.enabled:
            return

        self.gauges[name] = value
        self.metric_types[name] = MetricType.GAUGE

        if metadata and name not in self.metric_metadata:
            self.metric_metadata[name] = metadata

    def histogram(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Add a value to a histogram metric

        Args:
            name: Name of the histogram
            value: Value to add
            metadata: Optional metadata for this metric
        """
        if not self.enabled:
            return

        if name not in self.histograms:
            self.histograms[name] = []
            self.metric_types[name] = MetricType.HISTOGRAM
            if metadata:
                self.metric_metadata[name] = metadata

        self.histograms[name].append(value)

        # Keep histogram size reasonable
        if len(self.histograms[name]) > self.max_items_per_metric:
            self.histograms[name] = self.histograms[name][-self.max_items_per_metric :]

    def timing(self, name: str, value_ms: float, metadata: Dict[str, Any] = None):
        """
        Record a timing metric

        Args:
            name: Name of the timing
            value_ms: Value in milliseconds
            metadata: Optional metadata for this metric
        """
        if not self.enabled:
            return

        if name not in self.timings:
            self.timings[name] = []
            self.metric_types[name] = MetricType.TIMING
            if metadata:
                self.metric_metadata[name] = metadata

        self.timings[name].append(value_ms)

        # Keep timing size reasonable
        if len(self.timings[name]) > self.max_items_per_metric:
            self.timings[name] = self.timings[name][-self.max_items_per_metric :]

    def event(self, name: str, data: Dict[str, Any] = None):
        """
        Record an event with timestamp

        Args:
            name: Name of the event
            data: Additional data for the event
        """
        if not self.enabled:
            return

        if name not in self.events:
            self.events[name] = []
            self.metric_types[name] = MetricType.EVENT

        event_data = {"timestamp": time.time(), "datetime": datetime.datetime.now().isoformat(), "data": data or {}}

        self.events[name].append(event_data)

        # Keep events size reasonable
        if len(self.events[name]) > self.max_items_per_metric:
            self.events[name] = self.events[name][-self.max_items_per_metric :]

    def labeled_metric(self, name: str, labels: Dict[str, str], value: float):
        """
        Record a metric with labels

        Args:
            name: Name of the metric
            labels: Dictionary of labels
            value: Value to record
        """
        if not self.enabled:
            return

        if name not in self.labeled_metrics:
            self.labeled_metrics[name] = {}
            self.metric_types[name] = MetricType.LABELED

        # Convert labels to a string key
        label_key = json.dumps(labels, sort_keys=True)
        self.labeled_metrics[name][label_key] = value

    def create_timer(self, name: str) -> "Timer":
        """
        Create a timer that can be used as a context manager

        Args:
            name: Name of the timing metric

        Returns:
            Timer context manager
        """
        return Timer(self, name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics in a structured format

        Returns:
            Dictionary with all metrics data
        """
        return {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                    "recent": v[-10:] if v else [],
                    "median": statistics.median(v) if v else 0,
                    "stddev": statistics.stdev(v) if len(v) > 1 else 0,
                }
                for k, v in self.histograms.items()
            },
            "timings": {
                k: {
                    "count": len(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                    "recent": v[-10:] if v else [],
                    "p50": self._percentile(v, 50) if v else 0,
                    "p90": self._percentile(v, 90) if v else 0,
                    "p95": self._percentile(v, 95) if v else 0,
                    "p99": self._percentile(v, 99) if v else 0,
                }
                for k, v in self.timings.items()
            },
            "events": {k: v[-10:] if v else [] for k, v in self.events.items()},
            "labeled_metrics": {
                k: {json.loads(label_key): value for label_key, value in v.items()}
                for k, v in self.labeled_metrics.items()
            },
            "metadata": {
                "metric_count": len(self.metric_types),
                "metric_types": {k: v.name for k, v in self.metric_types.items()},
                "metric_metadata": self.metric_metadata,
            },
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """
        Calculate percentile of a list of values

        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = int(k)
        c = int(k) + 1 if k < len(sorted_values) - 1 else int(k)

        if f == c:
            return float(sorted_values[f])
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def flush(self):
        """Flush metrics to disk with improved organization"""
        if not self.enabled:
            return

        timestamp = int(time.time())
        date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        # Create date directory if it doesn't exist
        date_dir = self.metrics_dir / date_str
        date_dir.mkdir(exist_ok=True)

        metrics = self.get_all_metrics()

        # Save combined metrics file
        metrics_file = date_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save individual metrics by type for easier processing
        for metric_type in MetricType:
            metric_type_lower = metric_type.name.lower()
            type_dir = date_dir / metric_type_lower
            type_dir.mkdir(exist_ok=True)

            if metric_type == MetricType.COUNTER:
                data = metrics["counters"]
            elif metric_type == MetricType.GAUGE:
                data = metrics["gauges"]
            elif metric_type == MetricType.HISTOGRAM:
                data = metrics["histograms"]
            elif metric_type == MetricType.TIMING:
                data = metrics["timings"]
            elif metric_type == MetricType.EVENT:
                data = metrics["events"]
            elif metric_type == MetricType.LABELED:
                data = metrics["labeled_metrics"]

            if data:
                type_file = type_dir / f"metrics_{timestamp}.json"
                with open(type_file, "w") as f:
                    json.dump(data, f, indent=2)

        self.last_flush_time = time.time()

        # Clean up old metrics files
        self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """Clean up old metrics files based on retention policy"""
        current_time = time.time()
        max_age = self.retention_days * 24 * 60 * 60  # Convert days to seconds

        for path in self.metrics_dir.glob("**/*.json"):
            file_age = current_time - path.stat().st_mtime
            if file_age > max_age:
                try:
                    path.unlink()
                    logger.debug(f"Deleted old metrics file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old metrics file {path}: {str(e)}")

        # Cleanup empty directories
        for path in self.metrics_dir.glob("**/*"):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    path.rmdir()
                    logger.debug(f"Removed empty metrics directory: {path}")
                except Exception:
                    pass

    def reset(self):
        """Reset all metrics"""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timings.clear()
        self.events.clear()
        self.labeled_metrics.clear()

    def export_metrics_summary(self, format_type: str = "text", output_path: Optional[str] = None) -> str:
        """
        Export a summary of metrics

        Args:
            format_type: Format type ('text', 'json', 'html')
            output_path: Path to save summary

        Returns:
            Path to saved summary or summary string for text format
        """
        metrics = self.get_all_metrics()

        if format_type == "json":
            content = json.dumps(metrics, indent=2)

            if output_path:
                with open(output_path, "w") as f:
                    f.write(content)
                return output_path

            return content

        elif format_type == "html":
            if not output_path:
                output_path = f"metrics_summary_{int(time.time())}.html"

            with open(output_path, "w") as f:
                f.write(
                    """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Metrics Summary</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #333; }
                        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                        th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                        tr:nth-child(even) { background-color: #f2f2f2; }
                        th { background-color: #4CAF50; color: white; }
                    </style>
                </head>
                <body>
                    <h1>Metrics Summary</h1>
                    <p>Generated at: """
                    + metrics["datetime"]
                    + """</p>
                """
                )

                # Counters
                if metrics["counters"]:
                    f.write("<h2>Counters</h2>")
                    f.write("<table><tr><th>Name</th><th>Value</th></tr>")
                    for name, value in metrics["counters"].items():
                        f.write(f"<tr><td>{name}</td><td>{value}</td></tr>")
                    f.write("</table>")

                # Gauges
                if metrics["gauges"]:
                    f.write("<h2>Gauges</h2>")
                    f.write("<table><tr><th>Name</th><th>Value</th></tr>")
                    for name, value in metrics["gauges"].items():
                        f.write(f"<tr><td>{name}</td><td>{value}</td></tr>")
                    f.write("</table>")

                # Histograms
                if metrics["histograms"]:
                    f.write("<h2>Histograms</h2>")
                    f.write(
                        "<table><tr><th>Name</th><th>Count</th><th>Min</th><th>Avg</th><th>Max</th><th>Median</th><th>StdDev</th></tr>"
                    )
                    for name, data in metrics["histograms"].items():
                        f.write(f"<tr><td>{name}</td><td>{data['count']}</td><td>{data['min']:.2f}</td>")
                        f.write(f"<td>{data['avg']:.2f}</td><td>{data['max']:.2f}</td>")
                        f.write(f"<td>{data['median']:.2f}</td><td>{data['stddev']:.2f}</td></tr>")
                    f.write("</table>")

                # Timings
                if metrics["timings"]:
                    f.write("<h2>Timings</h2>")
                    f.write("<table><tr><th>Name</th><th>Count</th><th>Min (ms)</th><th>Avg (ms)</th>")
                    f.write("<th>P90 (ms)</th><th>P99 (ms)</th><th>Max (ms)</th></tr>")
                    for name, data in metrics["timings"].items():
                        f.write(f"<tr><td>{name}</td><td>{data['count']}</td><td>{data['min']:.2f}</td>")
                        f.write(f"<td>{data['avg']:.2f}</td><td>{data['p90']:.2f}</td>")
                        f.write(f"<td>{data['p99']:.2f}</td><td>{data['max']:.2f}</td></tr>")
                    f.write("</table>")

                # Events
                if metrics["events"]:
                    f.write("<h2>Recent Events</h2>")
                    for name, events in metrics["events"].items():
                        f.write(f"<h3>{name}</h3>")
                        f.write("<table><tr><th>Time</th><th>Data</th></tr>")
                        for event in events:
                            f.write(f"<tr><td>{event['datetime']}</td><td>{json.dumps(event['data'])}</td></tr>")
                        f.write("</table>")

                f.write("</body></html>")

            return output_path

        else:  # text format
            lines = []
            lines.append(f"Metrics Summary (Generated at {metrics['datetime']})")
            lines.append("=" * 80)

            # Counters
            if metrics["counters"]:
                lines.append("\nCounters:")
                for name, value in metrics["counters"].items():
                    lines.append(f"  {name}: {value}")

            # Gauges
            if metrics["gauges"]:
                lines.append("\nGauges:")
                for name, value in metrics["gauges"].items():
                    lines.append(f"  {name}: {value}")

            # Histograms
            if metrics["histograms"]:
                lines.append("\nHistograms:")
                for name, data in metrics["histograms"].items():
                    lines.append(f"  {name}:")
                    lines.append(f"    Count: {data['count']}")
                    lines.append(f"    Min: {data['min']:.2f}, Avg: {data['avg']:.2f}, Max: {data['max']:.2f}")
                    lines.append(f"    Median: {data['median']:.2f}, StdDev: {data['stddev']:.2f}")

            # Timings
            if metrics["timings"]:
                lines.append("\nTimings:")
                for name, data in metrics["timings"].items():
                    lines.append(f"  {name}:")
                    lines.append(f"    Count: {data['count']}")
                    lines.append(f"    Min: {data['min']:.2f}ms, Avg: {data['avg']:.2f}ms, Max: {data['max']:.2f}ms")
                    lines.append(f"    P50: {data['p50']:.2f}ms, P90: {data['p90']:.2f}ms, P99: {data['p99']:.2f}ms")

            content = "\n".join(lines)

            if output_path:
                with open(output_path, "w") as f:
                    f.write(content)
                return output_path

            return content

    def record_step_performance(self, step_id: str, duration: float, memory: float) -> None:
        """Record step performance for future predictions"""
        if step_id not in self.performance_predictions:
            self.performance_predictions[step_id] = {
                'count': 0,
                'total_duration': 0,
                'total_memory': 0
            }
            
        stats = self.performance_predictions[step_id]
        stats['count'] += 1
        stats['total_duration'] += duration
        stats['total_memory'] += memory
        
    def get_step_performance(self, step_id: str) -> Optional[Dict[str, float]]:
        """Get performance predictions for a step"""
        if step_id in self.performance_predictions:
            stats = self.performance_predictions[step_id]
            return {
                'avg_duration': stats['total_duration'] / stats['count'],
                'avg_memory': stats['total_memory'] / stats['count']
            }
        return None


# Timer context manager for easy timing
class Timer:
    """Context manager for easy timing of code blocks"""

    def __init__(self, metrics_collector: MetricsCollector, name: str):
        """
        Initialize timer

        Args:
            metrics_collector: Metrics collector to record timing
            name: Name of timing metric
        """
        self.metrics_collector = metrics_collector
        self.name = name
        self.start_time = None
        self.id = str(uuid.uuid4())

    def __enter__(self):
        """Start timing"""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric"""
        if self.start_time is not None:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.metrics_collector.timing(self.name, elapsed_ms)

            # If exception occurred, also record as event
            if exc_type is not None:
                self.metrics_collector.event(
                    f"error.{self.name}",
                    {
                        "exception_type": exc_type.__name__,
                        "exception_message": str(exc_val),
                        "timing_ms": elapsed_ms,
                        "timer_id": self.id,
                    },
                )
