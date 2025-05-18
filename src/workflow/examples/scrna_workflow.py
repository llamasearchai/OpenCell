import logging
from pathlib import Path
from typing import Any, Dict

from ...core.transcriptomics.sc_rna_processor import ScRNAProcessor
from ..pipeline.workflow_context import WorkflowContext
from ..pipeline.workflow_utils import workflow_step

logger = logging.getLogger(__name__)


@workflow_step(
    name="Load Data", description="Load scRNA-seq data from file", timeout_seconds=600, estimated_memory_mb=2000
)
async def load_data(context: WorkflowContext, data_path: str, format_type: str = "10x") -> Dict[str, Any]:
    """
    Load scRNA-seq data from file

    Args:
        context: Workflow context
        data_path: Path to data file or directory
        format_type: Format of the data

    Returns:
        Dictionary with loaded data info
    """
    logger.info(f"Loading {format_type} data from {data_path}")

    # Create processor
    processor = ScRNAProcessor()

    # Load data
    adata = processor.load_data(data_path, format_type=format_type)

    # Store processor in context for later
    context.add_result("processor", processor)
    context.add_result("adata", adata)

    return {"n_cells": adata.n_obs, "n_genes": adata.n_vars, "data_format": format_type}


@workflow_step(
    name="Preprocess Data",
    description="Perform quality control and preprocessing of scRNA-seq data",
    timeout_seconds=1800,
    estimated_memory_mb=4000,
)
async def preprocess_data(
    context: WorkflowContext, min_genes: int = 200, min_cells: int = 3, max_pct_mito: float = 20.0
) -> Dict[str, Any]:
    """
    Preprocess scRNA-seq data

    Args:
        context: Workflow context
        min_genes: Minimum genes per cell
        min_cells: Minimum cells per gene
        max_pct_mito: Maximum percent mitochondrial genes

    Returns:
        Dictionary with preprocessing results
    """
    logger.info("Preprocessing scRNA-seq data")

    # Get processor from context
    processor = context.get_result("processor")

    # Update parameters
    processor.parameters.min_genes = min_genes
    processor.parameters.min_cells = min_cells
    processor.parameters.max_pct_mito = max_pct_mito

    # Preprocess data
    adata = processor.preprocess()

    # Store updated adata in context
    context.add_result("adata", adata)

    return {
        "n_cells_after_qc": adata.n_obs,
        "n_genes_after_qc": adata.n_vars,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "max_pct_mito": max_pct_mito,
    }


@workflow_step(
    name="Run PCA", description="Perform PCA dimensionality reduction", timeout_seconds=600, estimated_memory_mb=4000
)
async def run_pca(context: WorkflowContext, n_pcs: int = 50) -> Dict[str, Any]:
    """
    Run PCA on preprocessed data

    Args:
        context: Workflow context
        n_pcs: Number of principal components

    Returns:
        Dictionary with PCA results
    """
    logger.info(f"Running PCA with {n_pcs} components")

    # Get processor from context
    processor = context.get_result("processor")

    # Update parameters
    processor.parameters.n_pcs = n_pcs

    # Run PCA
    processor.run_pca()

    # Get variance explained
    variance_explained = processor.results.get("variance_explained", [])
    top_variance = variance_explained[0] if len(variance_explained) > 0 else 0

    return {
        "n_pcs": n_pcs,
        "top_variance_explained": float(top_variance),
        "variance_explained": [float(v) for v in variance_explained[:10]],
    }


@workflow_step(
    name="Run Clustering", description="Perform graph-based clustering", timeout_seconds=900, estimated_memory_mb=4000
)
async def run_clustering(
    context: WorkflowContext, resolution: float = 0.8, clustering_method: str = "leiden"
) -> Dict[str, Any]:
    """
    Run clustering on dimensional reduction

    Args:
        context: Workflow context
        resolution: Resolution parameter for clustering
        clustering_method: Method for clustering (leiden or louvain)

    Returns:
        Dictionary with clustering results
    """
    logger.info(f"Running {clustering_method} clustering with resolution {resolution}")

    # Get processor from context
    processor = context.get_result("processor")

    # Update parameters
    processor.parameters.resolution = resolution
    processor.parameters.clustering_method = clustering_method

    # Run clustering
    processor.run_clustering()

    # Get clustering results
    n_clusters = processor.results.get("n_clusters", 0)

    return {"clustering_method": clustering_method, "resolution": resolution, "n_clusters": n_clusters}


@workflow_step(
    name="Run UMAP", description="Perform UMAP visualization", timeout_seconds=600, estimated_memory_mb=4000
)
async def run_umap(context: WorkflowContext) -> Dict[str, Any]:
    """
    Run UMAP for visualization

    Args:
        context: Workflow context

    Returns:
        Dictionary with UMAP results
    """
    logger.info("Running UMAP")

    # Get processor from context
    processor = context.get_result("processor")

    # Run UMAP
    processor.run_umap()

    return {"visualization": "umap"}


@workflow_step(
    name="Find Markers",
    description="Identify marker genes for clusters",
    timeout_seconds=1200,
    estimated_memory_mb=4000,
)
async def find_markers(context: WorkflowContext) -> Dict[str, Any]:
    """
    Find marker genes for clusters

    Args:
        context: Workflow context

    Returns:
        Dictionary with marker gene results
    """
    logger.info("Finding marker genes")

    # Get processor from context
    processor = context.get_result("processor")

    # Find markers
    markers_df = processor.find_markers()

    # Get top markers for each cluster
    top_markers = {}
    for group in markers_df["group"].unique():
        group_markers = markers_df[markers_df["group"] == group].sort_values("scores", ascending=False)
        top_markers[group] = group_markers["names"].tolist()[:10]

    # Store markers in context
    context.add_result("markers", top_markers)

    return {"n_markers": len(markers_df), "top_markers": top_markers}


@workflow_step(name="Save Results", description="Save analysis results", timeout_seconds=900, estimated_memory_mb=2000)
async def save_results(context: WorkflowContext, output_dir: str) -> Dict[str, Any]:
    """
    Save analysis results

    Args:
        context: Workflow context
        output_dir: Directory to save results

    Returns:
        Dictionary with paths to saved results
    """
    logger.info(f"Saving results to {output_dir}")

    # Get processor from context
    processor = context.get_result("processor")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate dataset ID from workflow run ID
    dataset_id = context.run_id

    # Normalize data
    # await processor.normalize_data(method="log1p") # Example: using log1p, others: scran, pearson_residuals, sctransform

    # Save results
    # metadata = await processor.save_results(dataset_id, str(output_path)) # F841: metadata not used
    await processor.save_results(dataset_id, str(output_path))

    # Add artifacts to context
    context.add_artifact("normalized_data_path", str(output_path / "normalized_data.h5ad"))
    context.add_artifact("h5ad_file", str(output_path / f"{dataset_id}.h5ad"))
    context.add_artifact("umap_plot", str(output_path / "figures" / f"_clusters_{dataset_id}.pdf"))

    return {"output_dir": str(output_path), "dataset_id": dataset_id, "artifacts": context.artifacts}
