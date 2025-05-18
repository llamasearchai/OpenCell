import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import psutil


class StepStatus(Enum):
    """Status of a workflow step"""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()
    SKIPPED = auto()


class ExecutionMode(Enum):
    """Execution mode for workflows"""

    SEQUENTIAL = auto()  # Execute steps one after another
    PARALLEL = auto()  # Execute steps in parallel where possible
    DISTRIBUTED = auto()  # Execute steps across multiple nodes


@dataclass
class WorkflowContext:
    """Context for a workflow execution"""

    workflow_id: str
    run_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    status: str = "PENDING"
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_statuses: Dict[str, StepStatus] = field(default_factory=dict)
    step_times: Dict[str, Dict[str, float]] = field(default_factory=dict)
    resource_usage: Dict[str, List[float]] = field(
        default_factory=lambda: {"cpu_percent": [], "memory_mb": [], "timestamp": []}
    )
    # Enhanced fields for better tracking
    progress: float = 0.0  # Overall progress from 0-1
    step_progress: Dict[str, float] = field(default_factory=dict)  # Progress of each step from 0-1
    warning_logs: List[str] = field(default_factory=list)  # Separate warnings from regular logs
    error_logs: List[str] = field(default_factory=list)  # Separate errors from regular logs
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # Performance metrics

    def add_result(self, key: str, value: Any) -> None:
        """Add a result to the context"""
        self.results[key] = value

    def add_artifact(self, key: str, path: str) -> None:
        """Add an artifact path to the context"""
        self.artifacts[key] = path

    def get_result(self, key: str, default: Any = None) -> Any:
        """Get a result from the context"""
        return self.results.get(key, default)

    def get_artifact(self, key: str, default: str = None) -> str:
        """Get an artifact path from the context"""
        return self.artifacts.get(key, default)

    def add_log(self, message: str) -> None:
        """Add a log message to the context"""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message to the context"""
        timestamp = datetime.now().isoformat()
        warning = f"[{timestamp}] WARNING: {message}"
        self.warning_logs.append(warning)
        self.logs.append(warning)

    def add_error(self, message: str) -> None:
        """Add an error message to the context"""
        timestamp = datetime.now().isoformat()
        error = f"[{timestamp}] ERROR: {message}"
        self.error_logs.append(error)
        self.logs.append(error)

    def update_resource_usage(self) -> None:
        """Update resource usage statistics"""
        process = psutil.Process(os.getpid())

        # Get CPU and memory usage
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Add to resource usage history
        self.resource_usage["cpu_percent"].append(cpu_percent)
        self.resource_usage["memory_mb"].append(memory_mb)
        self.resource_usage["timestamp"].append(time.time())

        # Keep only the last 60 measurements
        max_history = 60
        if len(self.resource_usage["timestamp"]) > max_history:
            self.resource_usage["cpu_percent"] = self.resource_usage["cpu_percent"][-max_history:]
            self.resource_usage["memory_mb"] = self.resource_usage["memory_mb"][-max_history:]
            self.resource_usage["timestamp"] = self.resource_usage["timestamp"][-max_history:]

    def update_step_progress(self, step_id: str, progress: float) -> None:
        """
        Update progress for a specific step

        Args:
            step_id: ID of the step
            progress: Progress value between 0 and 1
        """
        progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        self.step_progress[step_id] = progress

        # Update overall progress based on all steps
        if self.step_statuses:
            completed_steps = sum(
                1 for status in self.step_statuses.values() if status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            )
            running_steps = sum(1 for status in self.step_statuses.values() if status == StepStatus.RUNNING)

            if running_steps > 0:
                # Calculate progress including partially completed running steps
                running_progress = sum(
                    self.step_progress.get(step_id, 0.0)
                    for step_id, status in self.step_statuses.items()
                    if status == StepStatus.RUNNING
                )

                self.progress = (completed_steps + running_progress) / len(self.step_statuses)
            else:
                # Just use completed steps if no running steps
                self.progress = completed_steps / len(self.step_statuses) if self.step_statuses else 0.0

    def add_performance_metric(self, name: str, value: Any) -> None:
        """Add a performance metric to the context"""
        self.performance_metrics[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "parameters": self.parameters,
            "results": self.results,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "logs": self.logs,
            "status": self.status,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "step_statuses": {k: v.name for k, v in self.step_statuses.items()},
            "step_times": self.step_times,
            "resource_usage": self.resource_usage,
            "progress": self.progress,
            "step_progress": self.step_progress,
            "warning_logs": self.warning_logs,
            "error_logs": self.error_logs,
            "performance_metrics": self.performance_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Create context from dictionary"""
        # Convert string step statuses back to enum
        step_statuses = {}
        for step_id, status_str in data.get("step_statuses", {}).items():
            try:
                step_statuses[step_id] = StepStatus[status_str]
            except KeyError:
                step_statuses[step_id] = StepStatus.FAILED

        # Convert ISO strings to datetime objects
        start_time = None
        if data.get("start_time"):
            try:
                start_time = datetime.fromisoformat(data["start_time"])
            except ValueError:
                pass

        end_time = None
        if data.get("end_time"):
            try:
                end_time = datetime.fromisoformat(data["end_time"])
            except ValueError:
                pass

        # Create the context object with basic fields
        context = cls(
            workflow_id=data["workflow_id"],
            run_id=data["run_id"],
            parameters=data.get("parameters", {}),
            results=data.get("results", {}),
            artifacts=data.get("artifacts", {}),
            metadata=data.get("metadata", {}),
            logs=data.get("logs", []),
            status=data.get("status", "PENDING"),
            error=data.get("error"),
            start_time=start_time,
            end_time=end_time,
            step_statuses=step_statuses,
            step_times=data.get("step_times", {}),
            resource_usage=data.get("resource_usage", {"cpu_percent": [], "memory_mb": [], "timestamp": []}),
        )

        # Set advanced fields if they exist
        context.progress = data.get("progress", 0.0)
        context.step_progress = data.get("step_progress", {})
        context.warning_logs = data.get("warning_logs", [])
        context.error_logs = data.get("error_logs", [])
        context.performance_metrics = data.get("performance_metrics", {})

        return context
