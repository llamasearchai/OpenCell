import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, Index
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import (  # Or use sqlalchemy.types.UUID for a generic one if not tied to PostgreSQL
    UUID,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class StatusEnum(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        """Returns True if the status is a terminal state (workflow/step cannot continue)."""
        return self in {
            StatusEnum.COMPLETED,
            StatusEnum.FAILED,
            StatusEnum.CANCELLED,
            StatusEnum.TIMEOUT,
            StatusEnum.SKIPPED,
        }

    @property
    def is_successful(self) -> bool:
        """Returns True if the status indicates a successful completion."""
        return self == StatusEnum.COMPLETED

    @property
    def has_failed_or_timed_out(self) -> bool:
        """Returns True if the status is FAILED or TIMEOUT."""
        return self in {StatusEnum.FAILED, StatusEnum.TIMEOUT}


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps to models."""

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class Workflow(Base, TimestampMixin):
    """Workflow definition"""

    __tablename__ = "workflows"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    step_count = Column(Integer, default=0)
    enabled = Column(Boolean, default=True, nullable=False)
    default_parameters = Column(JSON)  # Default parameters for this workflow
    tags = Column(JSON)  # e.g., ["scrna", "analysis", "production"]
    owner_id = Column(String(255))  # User or service that owns/created the workflow
    custom_metadata = Column(JSON)  # Any other metadata for the workflow

    # Relationships
    steps = relationship("WorkflowStepDefinition", back_populates="workflow", cascade="all, delete-orphan")
    runs = relationship("WorkflowRun", back_populates="workflow", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Workflow(id={self.id}, name='{self.name}', version='{self.version}')>"


class WorkflowStepDefinition(Base, TimestampMixin):
    """Defines a step within a workflow template."""

    __tablename__ = "workflow_step_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False, index=True)
    step_id_in_workflow = Column(
        String(255), nullable=False
    )  # User-defined ID of the step within the workflow (e.g., "load_data")
    name = Column(String(255), nullable=False)
    description = Column(Text)
    function_identifier = Column(String(512))  # e.g., "module.submodule.function_name"
    default_parameters = Column(JSON)  # Default parameters for this step
    dependencies = Column(JSON)  # List of step_id_in_workflow strings this step depends on
    retry_count = Column(Integer, default=0)
    timeout_seconds = Column(Integer)
    estimated_memory_mb = Column(Integer)
    estimated_duration_seconds = Column(Integer)
    priority = Column(Integer, default=0)
    use_gpu = Column(Boolean, default=False)

    workflow = relationship("Workflow", back_populates="steps")

    def __repr__(self):
        return f"<WorkflowStepDefinition(id={self.id}, name='{self.name}', workflow_id={self.workflow_id})>"


class WorkflowRun(Base, TimestampMixin):
    """Workflow run instance with enhanced performance tracking"""

    __tablename__ = "workflow_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"), nullable=False, index=True)
    status = Column(SAEnum(StatusEnum), default=StatusEnum.PENDING, nullable=False, index=True)
    parameters = Column(JSON)  # Parameters used for this specific run
    results = Column(JSON)  # Results of the workflow run
    artifacts = Column(JSON)  # Paths or references to artifacts produced
    logs_summary = Column(Text)  # A summary or link to detailed logs
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)  # Calculated duration
    error_message = Column(Text)
    error_traceback = Column(Text)
    triggered_by = Column(String(255))  # e.g., user ID, cron job name
    custom_metadata = Column(JSON)  # Any other metadata for the run (e.g., resource usage summary)

    # Enhanced performance metrics
    peak_memory_mb = Column(Float)
    average_cpu_percent = Column(Float)
    total_disk_io_mb = Column(Float)
    network_io_mb = Column(Float)
    gpu_utilization = Column(JSON)  # {gpu_id: {utilization: float, memory_used: float}}
    critical_path_duration = Column(Float)  # Duration of critical path steps

    # Add index for faster performance queries
    __table_args__ = (
        Index('ix_workflow_runs_started_at', 'started_at'),
        Index('ix_workflow_runs_duration', 'duration_seconds'),
    )

    workflow = relationship("Workflow", back_populates="runs")
    step_runs = relationship("WorkflowStepRun", back_populates="workflow_run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<WorkflowRun(id={self.id}, workflow_id={self.workflow_id}, status='{self.status.name}')>"


class WorkflowStepRun(Base, TimestampMixin):
    """Workflow step run instance with detailed resource tracking"""

    __tablename__ = "workflow_step_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_run_id = Column(UUID(as_uuid=True), ForeignKey("workflow_runs.id"), nullable=False, index=True)
    step_definition_id = Column(
        UUID(as_uuid=True), ForeignKey("workflow_step_definitions.id"), nullable=False
    )  # Link to the definition
    step_id_in_workflow = Column(
        String(255), nullable=False, index=True
    )  # Matches step_id_in_workflow from definition
    status = Column(SAEnum(StatusEnum), default=StatusEnum.PENDING, nullable=False, index=True)
    parameters_used = Column(JSON)  # Actual parameters used for this step run (after overrides)
    result = Column(JSON)
    artifacts_produced = Column(JSON)
    logs = Column(Text)  # Or link to log storage
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    error_message = Column(Text)
    error_traceback = Column(Text)
    retry_attempt = Column(Integer, default=0)
    host_machine = Column(String(255))  # Machine where the step ran (for distributed setups)

    # Enhanced resource tracking
    peak_memory_mb = Column(Float)
    average_cpu_percent = Column(Float)
    disk_io_mb = Column(Float)
    network_io_mb = Column(Float)
    gpu_usage = Column(JSON)  # {gpu_id: {utilization: float, memory_used: float}}
    io_wait_time = Column(Float)  # Time spent waiting for I/O

    # Add index for performance analysis
    __table_args__ = (
        Index('ix_step_runs_duration', 'duration_seconds'),
        Index('ix_step_runs_resource', 'peak_memory_mb', 'average_cpu_percent'),
    )

    workflow_run = relationship("WorkflowRun", back_populates="step_runs")
    # step_definition = relationship("WorkflowStepDefinition") # If needed for direct access

    def __repr__(self):
        return f"<WorkflowStepRun(id={self.id}, run_id={self.workflow_run_id}, step='{self.step_id_in_workflow}', status='{self.status.name}')>"


class WorkflowScheduledRun(Base):
    """Scheduled workflow run"""

    __tablename__ = "workflow_scheduled_run"

    id = Column(String, primary_key=True, index=True)
    workflow_id = Column(String, ForeignKey("workflow.id"), nullable=False)
    parameters = Column(JSON)
    schedule_time = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)  # SCHEDULED, RUNNING, COMPLETED, FAILED, CANCELLED
    created_at = Column(DateTime, default=datetime.now)
    executed_at = Column(DateTime)
    run_id = Column(String, ForeignKey("workflow_run.id"))  # ID of the actual run when executed


# You might also want models for Datasets, AnalysisResults, etc., if they are part of this system.
# For example:
class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    data_type = Column(String(100))  # e.g., "scRNA-seq", "genomic_variants"
    source_path = Column(Text)  # Path to raw data or URI
    custom_metadata = Column(JSON)  # e.g., sample info, experimental conditions


class AnalysisResult(Base, TimestampMixin):
    __tablename__ = "analysis_results"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_run_id = Column(UUID(as_uuid=True), ForeignKey("workflow_runs.id"))
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    name = Column(String(255))
    result_type = Column(String(100))  # e.g., "normalized_matrix", "marker_genes_table"
    storage_path = Column(Text)  # Path to stored result file/object
    custom_metadata = Column(JSON)

    # Add new fields for better result tracking
    file_size_mb = Column(Float)
    file_format = Column(String(50))
    checksum = Column(String(64))  # For data integrity verification
    compression_ratio = Column(Float)

    # Add relationship to step run
    step_run_id = Column(UUID(as_uuid=True), ForeignKey('workflow_step_runs.id'))
    step_run = relationship("WorkflowStepRun", backref="analysis_results")

    # workflow_run = relationship("WorkflowRun")
    # dataset = relationship("Dataset")
