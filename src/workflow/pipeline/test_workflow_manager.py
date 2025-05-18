import asyncio
import pytest
import uuid
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from workflow.pipeline.workflow_manager import (
    WorkflowManager, 
    WorkflowStepDefinition, 
    ExecutionMode, 
    WorkflowContext,
    StepExecutionStatus
)

# Mock function resolver for testing
def mock_function_resolver(identifier: str):
    async def mock_function(*args, **kwargs):
        return {"message": f"Executed {identifier}"}
    return mock_function

# Define test fixtures
@pytest.fixture
async def workflow_manager():
    """Create a workflow manager instance for testing"""
    # Create a temporary directory for workflow artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a workflow manager
        manager = WorkflowManager(
            execution_mode=ExecutionMode.PARALLEL,
            max_workers=2,
            workflow_dir=tmpdir,
            enable_caching=True,
            use_optimization=True
        )
        
        # Set the function resolver
        manager.set_function_resolver(mock_function_resolver)
        
        yield manager

@pytest.fixture
async def sample_workflow(workflow_manager):
    """Create a sample workflow for testing"""
    steps = [
        WorkflowStepDefinition(
            id="step1",
            name="Step 1",
            description="First step",
            function_identifier="function1",
            parameters={"param1": "value1"},
            dependencies=[]
        ),
        WorkflowStepDefinition(
            id="step2",
            name="Step 2",
            description="Second step",
            function_identifier="function2",
            parameters={"param2": "value2"},
            dependencies=["step1"]
        ),
        WorkflowStepDefinition(
            id="step3",
            name="Step 3",
            description="Third step",
            function_identifier="function3",
            parameters={"param3": "value3"},
            dependencies=["step1", "step2"]
        )
    ]
    
    # Register workflow
    workflow_db_id = await workflow_manager.register_workflow_definition(
        name="Test Workflow",
        steps_definitions=steps,
        version="1.0.0",
        description="Workflow for testing",
        default_parameters={"default_param": True},
        tags=["test", "workflow"]
    )
    
    return workflow_db_id

# Define test classes
class TestWorkflowManager:
    @pytest.mark.asyncio
    async def test_initialization(self, workflow_manager):
        """Test workflow manager initialization"""
        assert workflow_manager.execution_mode == ExecutionMode.PARALLEL
        assert workflow_manager.max_workers == 2
        assert workflow_manager.enable_caching is True
        assert workflow_manager.use_optimization is True

    @pytest.mark.asyncio
    async def test_workflow_registration(self, workflow_manager):
        """Test workflow registration"""
        steps = [
            WorkflowStepDefinition(
                id="step1",
                name="Step 1",
                description="First step",
                function_identifier="function1",
                parameters={},
                dependencies=[]
            )
        ]
        
        workflow_id = await workflow_manager.register_workflow_definition(
            name="Simple Workflow",
            steps_definitions=steps,
            version="1.0.0"
        )
        
        assert workflow_id is not None
        assert isinstance(workflow_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_manager, sample_workflow):
        """Test workflow execution"""
        # Create a workflow run
        run_id = await workflow_manager.create_workflow_run(
            workflow_identifier=sample_workflow,
            parameters={"test_param": True}
        )
        
        # Execute the workflow
        context = await workflow_manager.execute_workflow(run_id)
        
        # Verify the context
        assert context.status == StepExecutionStatus.COMPLETED.name
        assert len(context.results) == 3  # Should have results for all steps
        assert "step1" in context.results
        assert "step2" in context.results
        assert "step3" in context.results
        
        # Verify step results
        for step_id in ["step1", "step2", "step3"]:
            assert context.results[step_id]["message"] == f"Executed function{step_id[-1]}"
    
    @pytest.mark.asyncio
    async def test_workflow_status(self, workflow_manager, sample_workflow):
        """Test workflow status retrieval"""
        # Create and execute a workflow run
        run_id = await workflow_manager.create_workflow_run(
            workflow_identifier=sample_workflow,
            parameters={}
        )
        await workflow_manager.execute_workflow(run_id)
        
        # Get the workflow status
        status = await workflow_manager.get_workflow_status(run_id)
        
        # Verify the status
        assert status["status"] == "COMPLETED"
        assert status["progress"] == 100.0
        assert status["total_steps"] == 3
        assert status["completed_steps"] == 3
        
        # Verify step statuses
        for step_id in ["step1", "step2", "step3"]:
            assert step_id in status["step_statuses"]
            assert status["step_statuses"][step_id]["status"] == "COMPLETED"
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, workflow_manager, sample_workflow):
        """Test workflow metrics retrieval"""
        # Create and execute a workflow run
        run_id = await workflow_manager.create_workflow_run(
            workflow_identifier=sample_workflow,
            parameters={}
        )
        await workflow_manager.execute_workflow(run_id)
        
        # Get the workflow metrics
        metrics = await workflow_manager.get_workflow_metrics(run_id)
        
        # Verify the metrics
        assert metrics["workflow_id"] == str(sample_workflow)
        assert metrics["status"] == "COMPLETED"
        assert "total_steps_defined" in metrics
        assert metrics["total_steps_defined"] == 3
        assert metrics["completed_steps"] == 3
        assert metrics["failed_steps"] == 0
        
        # Verify step metrics
        for step_id in ["step1", "step2", "step3"]:
            assert step_id in metrics["step_metrics"]
            assert metrics["step_metrics"][step_id]["status"] == "COMPLETED"
    
    @pytest.mark.asyncio
    async def test_workflow_visualization(self, workflow_manager, sample_workflow):
        """Test workflow visualization generation"""
        # Generate visualization
        visualization = await workflow_manager.generate_workflow_visualization(
            sample_workflow, 
            output_format="json"
        )
        
        # Verify it's a valid JSON string by checking for expected content
        assert "Test Workflow" in visualization
        assert "step1" in visualization
        assert "step2" in visualization
        assert "step3" in visualization
    
    @pytest.mark.asyncio
    async def test_workflow_export_import(self, workflow_manager, sample_workflow):
        """Test workflow export and import"""
        # Export the workflow
        exported = await workflow_manager.export_workflow_definition(
            sample_workflow, 
            format="json"
        )
        
        # Import the workflow
        imported_id = await workflow_manager.import_workflow_definition(
            exported, 
            format="json"
        )
        
        # Verify the imported workflow
        assert imported_id is not None
        assert isinstance(imported_id, uuid.UUID)
        assert imported_id != sample_workflow  # Should be a new ID
        
        # Create a run of the imported workflow
        run_id = await workflow_manager.create_workflow_run(
            workflow_identifier=imported_id,
            parameters={}
        )
        
        # Execute the workflow
        context = await workflow_manager.execute_workflow(run_id)
        
        # Verify execution works
        assert context.status == StepExecutionStatus.COMPLETED.name
        assert len(context.results) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_integrity_validation(self, workflow_manager, sample_workflow):
        """Test workflow integrity validation"""
        # Validate the workflow
        validation = await workflow_manager.validate_workflow_integrity(sample_workflow)
        
        # Verify validation results
        assert validation["workflow_id"] == str(sample_workflow)
        assert validation["is_valid"] is True
        assert len(validation["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_optimization(self, workflow_manager, sample_workflow):
        """Test workflow optimization"""
        # Optimize the workflow
        optimization = await workflow_manager.optimize_workflow_execution_plan(sample_workflow)
        
        # Verify optimization results
        assert optimization["workflow_id"] == str(sample_workflow)
        assert optimization["total_steps"] == 3
        assert "critical_path" in optimization
        assert "parallelizable_groups" in optimization
        assert "recommendations" in optimization

# Define main runner
if __name__ == "__main__":
    # This allows running tests directly or via pytest
    pytest.main(["-xvs", __file__]) 