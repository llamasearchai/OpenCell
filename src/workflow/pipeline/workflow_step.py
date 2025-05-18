import traceback
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import psutil
import os

from .workflow_context import StepStatus


@dataclass
class WorkflowStep:
    """An enhanced step in a workflow with improved tracking and error handling"""

    id: str
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: int = 60
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Enhanced fields for better step management
    progress: float = 0.0  # Progress from 0 to 1
    current_retry: int = 0
    priority: int = 0  # Higher priority = execute first
    estimated_memory_mb: int = 0  # Estimated memory usage in MB
    use_gpu: bool = False  # Whether step requires GPU
    use_process: bool = False  # Whether to use separate process
    expected_duration: Optional[float] = None  # Expected duration in seconds

    # Cancellation support
    cancel_requested: bool = False
    cancelled_at: Optional[datetime] = None

    # Better error tracking
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None

    # Metadata tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Add new fields for resource prediction
    predicted_duration: Optional[float] = None
    predicted_memory: Optional[float] = None
    actual_duration: Optional[float] = None
    actual_memory: Optional[float] = None

    def add_log(self, message: str) -> None:
        """Add a log message to the step"""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def get_duration(self) -> Optional[float]:
        """Get the step duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def update_progress(self, progress: float) -> None:
        """Update step progress (0-1)"""
        self.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1

    def mark_started(self) -> None:
        """Mark the step as started"""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now()
        self.add_log("Step started")

    def mark_completed(self, result: Any = None) -> None:
        """Mark the step as completed"""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
        self.progress = 1.0
        duration = self.get_duration()
        self.add_log(f"Step completed in {duration:.2f}s" if duration else "Step completed")

    def mark_failed(self, error: Exception) -> None:
        """Enhanced error handling with detailed logging"""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now()
        self.error = str(error)
        self.error_type = type(error).__name__
        self.error_traceback = ''.join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        
        # Log the full error context
        error_context = {
            "step_id": self.id,
            "error": self.error,
            "error_type": self.error_type,
            "timestamp": self.completed_at.isoformat(),
            "parameters": self.parameters
        }
        self.add_log(f"STEP_FAILURE: {json.dumps(error_context, indent=2)}")

    def mark_cancelled(self) -> None:
        """Mark the step as cancelled"""
        self.status = StepStatus.CANCELLED
        self.completed_at = datetime.now()
        self.cancelled_at = datetime.now()
        self.add_log("Step was cancelled")

    def request_cancellation(self) -> None:
        """Request step cancellation"""
        self.cancel_requested = True
        self.add_log("Cancellation requested")

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested"""
        return self.cancel_requested

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the step"""
        self.metadata[key] = value

    def add_tag(self, tag: str) -> None:
        """Add a tag to the step"""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to serializable dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "status": self.status.name,
            "result": self.result,
            "error": self.error,
            "logs": self.logs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "current_retry": self.current_retry,
            "priority": self.priority,
            "estimated_memory_mb": self.estimated_memory_mb,
            "use_gpu": self.use_gpu,
            "use_process": self.use_process,
            "expected_duration": self.expected_duration,
            "cancel_requested": self.cancel_requested,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """Create a step from dictionary"""
        # Handle datetime fields
        started_at = None
        if data.get("started_at"):
            try:
                started_at = datetime.fromisoformat(data["started_at"])
            except ValueError:
                pass

        completed_at = None
        if data.get("completed_at"):
            try:
                completed_at = datetime.fromisoformat(data["completed_at"])
            except ValueError:
                pass

        cancelled_at = None
        if data.get("cancelled_at"):
            try:
                cancelled_at = datetime.fromisoformat(data["cancelled_at"])
            except ValueError:
                pass

        # Handle status
        try:
            status = StepStatus[data.get("status", "PENDING")]
        except KeyError:
            status = StepStatus.PENDING

        # Create step
        step = cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            function=None,  # Function cannot be serialized
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            status=status,
            logs=data.get("logs", []),
            started_at=started_at,
            completed_at=completed_at,
            result=data.get("result"),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            current_retry=data.get("current_retry", 0),
            priority=data.get("priority", 0),
            estimated_memory_mb=data.get("estimated_memory_mb", 0),
            use_gpu=data.get("use_gpu", False),
            use_process=data.get("use_process", False),
            expected_duration=data.get("expected_duration"),
            cancel_requested=data.get("cancel_requested", False),
            cancelled_at=cancelled_at,
            error_type=data.get("error_type"),
            error_traceback=data.get("error_traceback"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )

        return step

    def predict_resources(self, historical_data: Dict[str, Any]) -> None:
        """Predict resource requirements based on historical data"""
        if self.id in historical_data:
            stats = historical_data[self.id]
            self.predicted_duration = stats.get('avg_duration')
            self.predicted_memory = stats.get('avg_memory')
            
    def record_actual_resources(self) -> None:
        """Record actual resource usage after step completion"""
        if self.started_at and self.completed_at:
            self.actual_duration = (self.completed_at - self.started_at).total_seconds()
            
        # Get memory usage
        process = psutil.Process(os.getpid())
        self.actual_memory = process.memory_info().rss / (1024 ** 2)  # MB
