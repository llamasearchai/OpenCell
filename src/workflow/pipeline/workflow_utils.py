import importlib
import inspect
import json
import logging
import pkgutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx

from .workflow_context import StepStatus
from .workflow_step import WorkflowStep

logger = logging.getLogger(__name__)


def discover_workflow_functions(package_name: str) -> Dict[str, Callable]:
    """
    Discover workflow functions in a package

    Args:
        package_name: Name of the package to discover functions in

    Returns:
        Dictionary mapping function names to callables
    """
    functions = {}

    try:
        package = importlib.import_module(package_name)
    except ImportError:
        logger.error(f"Package {package_name} not found")
        return functions

    # Get the path to the package
    package_path = getattr(package, "__path__", [None])[0]
    if not package_path:
        logger.error(f"Could not determine path for package {package_name}")
        return functions

    # Walk through all modules in the package
    for _, module_name, is_pkg in pkgutil.walk_packages([package_path], f"{package_name}."):
        if is_pkg:
            continue

        try:
            module = importlib.import_module(module_name)

            # Find all functions in the module
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # Check if this is a workflow function (has workflow_step decorator or metadata)
                if hasattr(obj, "is_workflow_step") and obj.is_workflow_step:
                    function_id = f"{module_name}.{name}"
                    functions[function_id] = obj
                    logger.debug(f"Discovered workflow function: {function_id}")
        except ImportError as e:
            logger.error(f"Error importing module {module_name}: {str(e)}")

    return functions


def workflow_step(
    *,
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    retry_count: int = 0,
    use_gpu: bool = False,
    estimated_memory_mb: int = 0,
    priority: int = 0,
):
    """
    Decorator to mark a function as a workflow step

    Args:
        id: ID for the step (defaults to function name)
        name: Display name for the step
        description: Description of the step
        timeout_seconds: Maximum execution time in seconds
        retry_count: Number of times to retry on failure
        use_gpu: Whether the step requires a GPU
        estimated_memory_mb: Estimated memory usage in MB
        priority: Priority for execution ordering

    Returns:
        Decorated function
    """

    def decorator(func):
        # Store metadata on the function
        func.is_workflow_step = True
        func.step_id = id or func.__name__
        func.step_name = name or func.__name__
        func.step_description = description or func.__doc__ or ""
        func.timeout_seconds = timeout_seconds
        func.retry_count = retry_count
        func.use_gpu = use_gpu
        func.estimated_memory_mb = estimated_memory_mb
        func.priority = priority

        return func

    return decorator


def load_workflow_definition(path: str) -> Dict[str, Any]:
    """
    Load workflow definition from a file

    Args:
        path: Path to workflow definition file (JSON or YAML)

    Returns:
        Workflow definition as a dictionary
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Workflow definition file not found: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix.lower() in [".yaml", ".yml"]:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def build_workflow_graph(steps: List[WorkflowStep]) -> nx.DiGraph:
    """
    Build a directed graph of workflow steps

    Args:
        steps: List of workflow steps

    Returns:
        Directed acyclic graph of steps
    """
    # Create graph
    G = nx.DiGraph()

    # Add nodes for each step
    step_map = {step.id: step for step in steps}
    for step in steps:
        G.add_node(step.id, step=step)

    # Add edges for dependencies
    for step in steps:
        for dep_id in step.dependencies:
            if dep_id not in step_map:
                logger.warning(f"Step {step.id} depends on unknown step {dep_id}")
                continue

            G.add_edge(dep_id, step.id)

    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            raise ValueError(f"Workflow contains cycles: {cycles}")
    except nx.NetworkXNoCycle:
        pass

    return G


def visualize_workflow(graph: nx.DiGraph, output_path: Optional[str] = None) -> str:
    """
    Visualize a workflow graph

    Args:
        graph: Directed acyclic graph of steps
        output_path: Path to save visualization

    Returns:
        Path to saved visualization
    """
    try:
        import matplotlib.pyplot as plt

        # Create figure
        plt.figure(figsize=(12, 8))

        # Get node positions using hierarchical layout
        pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")

        # Get node colors based on status
        node_colors = []
        for node in graph.nodes():
            step = graph.nodes[node]["step"]
            if step.status == StepStatus.COMPLETED:
                node_colors.append("lightgreen")
            elif step.status == StepStatus.RUNNING:
                node_colors.append("lightblue")
            elif step.status == StepStatus.FAILED:
                node_colors.append("salmon")
            elif step.status == StepStatus.PENDING:
                node_colors.append("lightgray")
            else:
                node_colors.append("white")

        # Draw nodes and edges
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.5, arrowsize=20)

        # Add labels
        labels = {node: f"{graph.nodes[node]['step'].name}\n({node})" for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)

        # Set layout
        plt.axis("off")
        plt.tight_layout()

        # Save or show
        if output_path:
            plt.savefig(output_path)
            return output_path
        else:
            temp_path = f"/tmp/workflow_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(temp_path)
            return temp_path

    except ImportError:
        logger.warning("Matplotlib or PyGraphviz not available. Visualization skipped.")
        return ""
