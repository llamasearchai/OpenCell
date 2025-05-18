import logging
from typing import Any, Dict, Optional

from ..pipeline.workflow_factory import WorkflowFactory
from .scrna_workflow import find_markers, load_data, preprocess_data, run_clustering, run_pca, run_umap, save_results

logger = logging.getLogger(__name__)


def create_scrna_workflow(workflow_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a scRNA-seq analysis workflow

    Args:
        workflow_id: Optional workflow ID

    Returns:
        Workflow definition
    """
    # Create workflow factory
    factory = WorkflowFactory()

    # Create workflow steps
    steps = [
        {
            "id": "load_data",
            "name": "Load Data",
            "function": load_data,
            "parameters": {"format_type": "10x"},
            "dependencies": [],
        },
        {
            "id": "preprocess",
            "name": "Preprocess Data",
            "function": preprocess_data,
            "parameters": {"min_genes": 200, "min_cells": 3, "max_pct_mito": 20.0},
            "dependencies": ["load_data"],
        },
        {
            "id": "run_pca",
            "name": "Run PCA",
            "function": run_pca,
            "parameters": {"n_pcs": 50},
            "dependencies": ["preprocess"],
        },
        {
            "id": "run_clustering",
            "name": "Run Clustering",
            "function": run_clustering,
            "parameters": {"resolution": 0.8, "clustering_method": "leiden"},
            "dependencies": ["run_pca"],
        },
        {"id": "run_umap", "name": "Run UMAP", "function": run_umap, "dependencies": ["run_clustering"]},
        {"id": "find_markers", "name": "Find Markers", "function": find_markers, "dependencies": ["run_clustering"]},
        {
            "id": "save_results",
            "name": "Save Results",
            "function": save_results,
            "parameters": {"output_dir": "/data/results"},
            "dependencies": ["run_umap", "find_markers"],
        },
    ]

    # Create workflow
    workflow = factory.create_workflow_from_functions(
        name="scRNA-seq Analysis",
        steps=steps,
        workflow_id=workflow_id or "scrna_seq_analysis",
        description="Complete scRNA-seq analysis workflow",
        version="1.0.0",
        execution_mode="PARALLEL",
    )

    return workflow
