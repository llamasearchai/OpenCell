import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Any, List

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow.pipeline.workflow_manager import (
    WorkflowManager, 
    WorkflowStepDefinition, 
    ExecutionMode, 
    WorkflowContext
)
from workflow.database.db import setup_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define example workflow functions (these would normally be in a separate module)
async def load_data(context: WorkflowContext, data_path: str = "data/sample.csv") -> Dict[str, Any]:
    """Example step to load data"""
    logger.info(f"Loading data from {data_path}")
    # Simulate loading data
    await asyncio.sleep(1)
    return {"rows": 1000, "columns": 50}

async def preprocess_data(context: WorkflowContext, normalize: bool = True) -> Dict[str, Any]:
    """Example step to preprocess data"""
    data_info = context.get_result("load_data")
    if not data_info:
        raise ValueError("No data loaded. Make sure load_data step runs first.")
        
    logger.info(f"Preprocessing {data_info['rows']} rows. Normalize: {normalize}")
    # Simulate preprocessing
    await asyncio.sleep(2)
    return {"processed_rows": data_info["rows"], "normalized": normalize}

async def train_model(context: WorkflowContext, model_type: str = "linear", learning_rate: float = 0.01) -> Dict[str, Any]:
    """Example step to train a model"""
    preprocess_info = context.get_result("preprocess_data")
    if not preprocess_info:
        raise ValueError("No preprocessed data. Make sure preprocess_data step runs first.")
        
    logger.info(f"Training {model_type} model with learning rate {learning_rate}")
    # Simulate model training
    await asyncio.sleep(3)
    return {"model_type": model_type, "accuracy": 0.85, "parameters": 1000}

async def evaluate_model(context: WorkflowContext) -> Dict[str, Any]:
    """Example step to evaluate the model"""
    model_info = context.get_result("train_model")
    if not model_info:
        raise ValueError("No model trained. Make sure train_model step runs first.")
        
    logger.info(f"Evaluating {model_info['model_type']} model")
    # Simulate model evaluation
    await asyncio.sleep(1.5)
    return {
        "accuracy": model_info["accuracy"] * 0.98,  # Slight difference from training accuracy
        "precision": 0.82,
        "recall": 0.79
    }

async def save_model(context: WorkflowContext, output_path: str = "models/model.pkl") -> Dict[str, Any]:
    """Example step to save the model"""
    model_info = context.get_result("train_model")
    eval_info = context.get_result("evaluate_model")
    if not model_info or not eval_info:
        raise ValueError("Missing model training or evaluation results")
        
    logger.info(f"Saving model to {output_path}")
    # Simulate saving model
    await asyncio.sleep(0.5)
    return {"model_path": output_path, "test_accuracy": eval_info["accuracy"]}

# Function resolver
def resolve_function(identifier: str):
    """Resolve function identifier to actual function"""
    function_map = {
        "load_data": load_data,
        "preprocess_data": preprocess_data,
        "train_model": train_model,
        "evaluate_model": evaluate_model,
        "save_model": save_model
    }
    
    if identifier not in function_map:
        raise ValueError(f"Unknown function identifier: {identifier}")
    
    return function_map[identifier]

async def create_example_workflow(manager: WorkflowManager) -> str:
    """Create an example ML workflow definition"""
    # Define workflow steps
    steps = [
        WorkflowStepDefinition(
            id="load_data",
            name="Load Dataset",
            description="Load data from CSV file",
            function_identifier="load_data",
            parameters={"data_path": "data/example.csv"},
            dependencies=[],
            timeout_seconds=60,
            retry_count=2,
            priority=10
        ),
        WorkflowStepDefinition(
            id="preprocess_data",
            name="Preprocess Data",
            description="Clean and normalize data",
            function_identifier="preprocess_data",
            parameters={"normalize": True},
            dependencies=["load_data"],
            timeout_seconds=120,
            retry_count=1
        ),
        WorkflowStepDefinition(
            id="train_model",
            name="Train Model",
            description="Train machine learning model",
            function_identifier="train_model",
            parameters={"model_type": "random_forest", "learning_rate": 0.05},
            dependencies=["preprocess_data"],
            timeout_seconds=300,
            retry_count=0,
            use_gpu=True  # This step would use GPU if available
        ),
        WorkflowStepDefinition(
            id="evaluate_model",
            name="Evaluate Model",
            description="Evaluate model performance on test data",
            function_identifier="evaluate_model",
            parameters={},
            dependencies=["train_model"],
            timeout_seconds=60
        ),
        WorkflowStepDefinition(
            id="save_model",
            name="Save Model",
            description="Save trained model to disk",
            function_identifier="save_model",
            parameters={"output_path": "models/trained_model.pkl"},
            dependencies=["train_model", "evaluate_model"],
            timeout_seconds=60,
            retry_count=3
        )
    ]
    
    # Register workflow
    workflow_db_id = await manager.register_workflow_definition(
        name="ML Training Pipeline",
        steps_definitions=steps,
        version="1.0.0",
        description="Example machine learning training pipeline",
        default_parameters={"verbose": True},
        tags=["machine_learning", "example", "training"]
    )
    
    logger.info(f"Registered workflow with ID: {workflow_db_id}")
    return str(workflow_db_id)

async def run_workflow_test(manager: WorkflowManager, workflow_id: str):
    """Run the workflow and demonstrate features"""
    # Basic workflow execution
    logger.info("=== Running basic workflow execution ===")
    run_id = await manager.create_workflow_run(
        workflow_identifier=workflow_id,
        parameters={"verbose": True}
    )
    
    context = await manager.execute_workflow(run_id)
    logger.info(f"Workflow completed with status: {context.status}")
    
    # Get detailed status
    status = await manager.get_workflow_status(run_id)
    logger.info(f"Workflow status summary: {status['status']}, Progress: {status['progress']}%")
    
    # Get workflow metrics
    metrics = await manager.get_workflow_metrics(run_id)
    logger.info(f"Critical path duration: {metrics.get('critical_path_duration_seconds', 'N/A')}s")
    
    # Workflow visualization
    logger.info("=== Generating workflow visualization ===")
    visualization = await manager.generate_workflow_visualization(workflow_id, output_format="json")
    logger.info(f"Generated visualization (showing first 100 chars): {visualization[:100]}...")
    
    # Run comprehensive tests
    logger.info("=== Running workflow tests ===")
    test_parameters = [
        {"verbose": True},  # Default parameters
        {"data_path": "data/other_dataset.csv", "model_type": "neural_network"}  # Different parameters
    ]
    test_results = await manager.run_workflow_tests(workflow_id, test_parameters)
    logger.info(f"Tests completed: {test_results['tests_passed']}/{test_results['test_count']} passed")
    
    # Run benchmarks
    logger.info("=== Running workflow benchmark ===")
    benchmark_results = await manager.benchmark_workflow(workflow_id, iterations=2)
    logger.info(f"Benchmark summary: {benchmark_results['performance_stats']}")
    
    # Validate workflow integrity
    logger.info("=== Validating workflow integrity ===")
    validation_report = await manager.validate_workflow_integrity(workflow_id)
    logger.info(f"Workflow valid: {validation_report['is_valid']}, Issues: {len(validation_report['issues'])}, Warnings: {len(validation_report['warnings'])}")
    
    # Export workflow definition
    logger.info("=== Exporting workflow definition ===")
    workflow_json = await manager.export_workflow_definition(workflow_id, format="json")
    logger.info(f"Exported workflow JSON (showing first 100 chars): {workflow_json[:100]}...")
    
    # Run advanced optimizations
    logger.info("=== Running workflow optimizations ===")
    optimization_report = await manager.optimize_workflow_execution_plan(workflow_id)
    logger.info(f"Optimization recommendations: {len(optimization_report['recommendations'])}")
    logger.info(f"Estimated speedup: {optimization_report['estimated_speedup_percent']}%")
    
    # Generate monitoring dashboard
    logger.info("=== Generating monitoring dashboard ===")
    dashboard = await manager.generate_workflow_monitor_dashboard(run_id)
    logger.info(f"Generated monitoring dashboard (showing first 100 chars): {dashboard[:100]}...")
    
    return {
        "run_id": str(run_id),
        "status": context.status,
        "optimization_recommendations": len(optimization_report['recommendations']),
        "estimated_speedup": optimization_report['estimated_speedup_percent']
    }

async def main():
    """Main function to demonstrate the enhanced workflow manager"""
    logger.info("Initializing workflow manager...")
    
    # Setup database
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///workflow_test.db")
    await setup_database(db_url)
    
    # Initialize the workflow manager
    manager = WorkflowManager(
        execution_mode=ExecutionMode.PARALLEL,
        max_workers=4,
        workflow_dir="./workflow_artifacts",
        enable_caching=True,
        enable_prometheus=False,
        use_optimization=True
    )
    
    # Set function resolver
    manager.set_function_resolver(resolve_function)
    
    try:
        # Create workflow definition
        workflow_id = await create_example_workflow(manager)
        
        # Run tests and demonstrations
        results = await run_workflow_test(manager, workflow_id)
        
        logger.info(f"All tests completed successfully. Final results: {results}")
        
    except Exception as e:
        logger.error(f"Error during workflow demonstration: {str(e)}", exc_info=True)
    
    logger.info("Workflow manager demonstration completed.")

if __name__ == "__main__":
    asyncio.run(main()) 