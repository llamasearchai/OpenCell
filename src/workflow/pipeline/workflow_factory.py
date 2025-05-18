import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .workflow_step import WorkflowStep
from .workflow_utils import build_workflow_graph, discover_workflow_functions, load_workflow_definition

logger = logging.getLogger(__name__)


class WorkflowFactory:
    """Factory class for creating workflow definitions"""

    def __init__(self, functions_registry: Optional[Dict[str, Callable]] = None):
        """
        Initialize the workflow factory

        Args:
            functions_registry: Optional registry of workflow functions
        """
        self.functions_registry = functions_registry or {}

    def load_functions_from_package(self, package_name: str) -> None:
        """
        Load workflow functions from a package

        Args:
            package_name: Name of the package to load functions from
        """
        functions = discover_workflow_functions(package_name)
        self.functions_registry.update(functions)
        logger.info(f"Loaded {len(functions)} workflow functions from {package_name}")

    def create_workflow_from_definition(
        self, definition: Union[Dict[str, Any], str], workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow from a definition

        Args:
            definition: Workflow definition as a dictionary or path to definition file
            workflow_id: Optional ID for the workflow

        Returns:
            Workflow definition as a dictionary
        """
        # Load definition from file if string path is provided
        if isinstance(definition, str):
            definition = load_workflow_definition(definition)

        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = definition.get("id") or f"wf_{uuid.uuid4().hex[:8]}"

        # Create workflow dictionary
        workflow = {
            "id": workflow_id,
            "name": definition.get("name", "Unnamed Workflow"),
            "description": definition.get("description", ""),
            "version": definition.get("version", "1.0.0"),
            "execution_mode": definition.get("execution_mode", "PARALLEL"),
            "steps": [],
            "parameters": definition.get("parameters", {}),
        }

        # Create steps
        steps = []
        for step_def in definition.get("steps", []):
            step_id = step_def.get("id") or f"step_{uuid.uuid4().hex[:8]}"
            function_id = step_def.get("function")

            if not function_id:
                raise ValueError(f"Step {step_id} has no function defined")

            # Get function from registry
            function = self.functions_registry.get(function_id)
            if not function:
                raise ValueError(f"Function {function_id} not found in registry")

            # Create step
            step = WorkflowStep(
                id=step_id,
                name=step_def.get("name", function.__name__),
                description=step_def.get("description", getattr(function, "__doc__", "")),
                function=function,
                parameters=step_def.get("parameters", {}),
                dependencies=step_def.get("dependencies", []),
                timeout_seconds=step_def.get("timeout_seconds", getattr(function, "timeout_seconds", None)),
                retry_count=step_def.get("retry_count", getattr(function, "retry_count", 0)),
                retry_delay_seconds=step_def.get("retry_delay_seconds", 60),
                priority=step_def.get("priority", getattr(function, "priority", 0)),
                estimated_memory_mb=step_def.get("estimated_memory_mb", getattr(function, "estimated_memory_mb", 0)),
                use_gpu=step_def.get("use_gpu", getattr(function, "use_gpu", False)),
                use_process=step_def.get("use_process", False),
            )

            steps.append(step)

        # Validate workflow
        self._validate_workflow(steps)

        # Add steps to workflow
        workflow["steps"] = steps

        return workflow

    def create_workflow_from_functions(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        workflow_id: Optional[str] = None,
        description: str = "",
        version: str = "1.0.0",
        execution_mode: str = "PARALLEL",
    ) -> Dict[str, Any]:
        """
        Create a workflow from a list of functions and metadata

        Args:
            name: Name of the workflow
            steps: List of step definitions
            workflow_id: Optional ID for the workflow
            description: Optional description
            version: Optional version
            execution_mode: Execution mode (SEQUENTIAL, PARALLEL, DISTRIBUTED)

        Returns:
            Workflow definition as a dictionary
        """
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        # Create workflow dictionary
        workflow = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "version": version,
            "execution_mode": execution_mode,
            "steps": [],
            "parameters": {},
        }

        # Create steps
        workflow_steps = []
        for step_def in steps:
            step_id = step_def.get("id") or f"step_{uuid.uuid4().hex[:8]}"
            function = step_def.get("function")

            if not function:
                raise ValueError(f"Step {step_id} has no function defined")

            # If function is a string, get it from registry
            if isinstance(function, str):
                function_id = function
                function = self.functions_registry.get(function_id)
                if not function:
                    raise ValueError(f"Function {function_id} not found in registry")

            # Create step
            step = WorkflowStep(
                id=step_id,
                name=step_def.get("name", function.__name__),
                description=step_def.get("description", getattr(function, "__doc__", "")),
                function=function,
                parameters=step_def.get("parameters", {}),
                dependencies=step_def.get("dependencies", []),
                timeout_seconds=step_def.get("timeout_seconds", getattr(function, "timeout_seconds", None)),
                retry_count=step_def.get("retry_count", getattr(function, "retry_count", 0)),
                retry_delay_seconds=step_def.get("retry_delay_seconds", 60),
                priority=step_def.get("priority", getattr(function, "priority", 0)),
                estimated_memory_mb=step_def.get("estimated_memory_mb", getattr(function, "estimated_memory_mb", 0)),
                use_gpu=step_def.get("use_gpu", getattr(function, "use_gpu", False)),
                use_process=step_def.get("use_process", False),
            )

            workflow_steps.append(step)

        # Validate workflow
        self._validate_workflow(workflow_steps)

        # Add steps to workflow
        workflow["steps"] = workflow_steps

        return workflow

    def create_linear_workflow(
        self,
        name: str,
        functions: List[Union[Callable, str]],
        workflow_id: Optional[str] = None,
        description: str = "",
        version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """
        Create a linear workflow where each step depends on the previous one

        Args:
            name: Name of the workflow
            functions: List of functions to use as steps
            workflow_id: Optional ID for the workflow
            description: Optional description
            version: Optional version

        Returns:
            Workflow definition as a dictionary
        """
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

        # Create step definitions
        steps = []
        previous_step_id = None

        for i, func in enumerate(functions):
            # Handle function from registry if string
            if isinstance(func, str):
                function_id = func
                func = self.functions_registry.get(function_id)
                if not func:
                    raise ValueError(f"Function {function_id} not found in registry")

            step_id = f"step_{i+1}"

            # Define dependencies based on previous step
            dependencies = []
            if previous_step_id:
                dependencies.append(previous_step_id)

            # Create step definition
            step_def = {
                "id": step_id,
                "name": getattr(func, "__name__", f"Step {i+1}"),
                "function": func,
                "dependencies": dependencies,
            }

            steps.append(step_def)
            previous_step_id = step_id

        # Create workflow from step definitions
        return self.create_workflow_from_functions(
            name=name,
            steps=steps,
            workflow_id=workflow_id,
            description=description,
            version=version,
            execution_mode="SEQUENTIAL",
        )

    def save_workflow_definition(self, workflow: Dict[str, Any], output_path: str) -> str:
        """
        Save workflow definition to a file

        Args:
            workflow: Workflow definition as a dictionary
            output_path: Path to save definition to

        Returns:
            Path to saved definition
        """
        # Create serializable version of workflow
        serializable = {
            "id": workflow["id"],
            "name": workflow["name"],
            "description": workflow["description"],
            "version": workflow["version"],
            "execution_mode": workflow["execution_mode"],
            "parameters": workflow["parameters"],
            "steps": [],
        }

        # Convert steps to serializable format
        for step in workflow["steps"]:
            # Get function info
            if callable(step.function):
                function_module = step.function.__module__
                function_name = step.function.__name__
                function_id = f"{function_module}.{function_name}"
            else:
                function_id = str(step.function)

            # Create serializable step
            serializable_step = {
                "id": step.id,
                "name": step.name,
                "description": step.description,
                "function": function_id,
                "parameters": step.parameters,
                "dependencies": step.dependencies,
                "timeout_seconds": step.timeout_seconds,
                "retry_count": step.retry_count,
                "retry_delay_seconds": step.retry_delay_seconds,
                "priority": step.priority,
                "estimated_memory_mb": step.estimated_memory_mb,
                "use_gpu": step.use_gpu,
                "use_process": step.use_process,
            }

            serializable["steps"].append(serializable_step)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if output_path.suffix.lower() == ".json":
                json.dump(serializable, f, indent=2)
            elif output_path.suffix.lower() in [".yml", ".yaml"]:
                import yaml

                yaml.dump(serializable, f)
            else:
                # Default to JSON
                json.dump(serializable, f, indent=2)

        return str(output_path)

    def _validate_workflow(self, steps: List[WorkflowStep]) -> None:
        """
        Validate a workflow

        Args:
            steps: List of workflow steps

        Raises:
            ValueError: If workflow is invalid
        """
        # Check for duplicate step IDs
        step_ids = [step.id for step in steps]
        duplicates = {x for x in step_ids if step_ids.count(x) > 1}
        if duplicates:
            raise ValueError(f"Duplicate step IDs found: {duplicates}")

        # Check for missing dependencies
        step_id_set = set(step_ids)
        for step in steps:
            for dep_id in step.dependencies:
                if dep_id not in step_id_set:
                    raise ValueError(f"Step {step.id} depends on unknown step {dep_id}")

        # Check for cycles in dependency graph
        try:
            build_workflow_graph(steps)
        except ValueError as e:
            raise ValueError(f"Invalid workflow: {str(e)}") from e
