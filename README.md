# OpenCell

<img src="OpenCell.svg" alt="OpenCell Logo" width="200"/>

A high-performance single-cell RNA-seq analysis pipeline with GPU acceleration capabilities, designed for computational biologists and bioinformaticians working with large-scale genomic datasets.

*Developed by [Nik Jois](mailto:nikjois@llamasearch.ai)*

## Overview

OpenCell is a comprehensive toolkit for analyzing single-cell RNA sequencing data, built with a focus on performance, scalability, and flexibility. Our pipeline leverages GPU acceleration when available to significantly reduce computational time for large datasets, while also providing robust fallback mechanisms for systems without specialized hardware.

### Key Applications

- **Cancer Research**: Identify rare cell populations and analyze tumor heterogeneity
- **Immunology**: Characterize immune cell subtypes and responses
- **Developmental Biology**: Track cellular differentiation trajectories
- **Drug Discovery**: Analyze cellular responses to compounds

## Features

- **GPU-accelerated analysis** for faster processing of large datasets (3-4.5x speedup)
- **Robust dependency management** with graceful fallbacks for optional libraries
- **Comprehensive batch correction** with Harmony, Combat, and custom methods
- **Advanced cell filtering** with doublet detection and quality metrics
- **Flexible normalization** supporting multiple methods:
  - log1p (standard normalization)
  - scran (pooling-based size factors)
  - Pearson residuals (variance stabilization)
  - SCTransform (regression-based normalization via R)
- **Optimized clustering** with benchmark-tested parameter selection
- **Pathway analysis** integration with GSEA and GO enrichment
- **Trajectory inference** for developmental and differentiation studies

## Installation

```bash
# Clone the repository
git clone https://github.com/llamasearchai/OpenCell.git
cd OpenCell

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU dependencies (requires CUDA toolkit)
pip install -r requirements-gpu.txt
```

### Optional Dependencies

OpenCell supports multiple optional components that can be installed based on your specific needs:

- For **doublet detection**: `pip install scrublet`
- For **R integration**: `pip install rpy2` (also requires R installation with Seurat, scran, etc.)
- For **batch correction with Harmony**: `pip install harmonypy`
- For **pathway analysis**: `pip install gseapy goatools`

## Quick Start

```python
from src.core.transcriptomics.sc_rna_processor import ScRNAProcessor, ScRNAParameters

# Configure analysis parameters
params = ScRNAParameters(
    min_genes=200,
    min_cells=3,
    max_pct_mito=20.0,
    n_pcs=50,
    resolution=0.5,
    use_gpu=True,  # Set to False if no GPU is available
    normalization_method="log1p",  # Options: "log1p", "scran", "pearson_residuals", "sctransform"
    batch_correction=True,
    batch_key="sample_id"  # Column in your metadata that defines batches
)

# Initialize processor
processor = ScRNAProcessor(parameters=params)

# Load and analyze data
processor.load_data("path/to/data", format_type="10x")
adata = processor.run_analysis()

# Access results
clusters = adata.obs["leiden"]  # or "louvain" depending on clustering method
umap_coords = adata.obsm["X_umap"]
marker_genes = processor.find_markers()

# Save results
processor.save_results(
    "output_directory",
    save_adata=True,
    save_plots=True,
    export_csv=True,
    db_file="results.db"  # Optional SQLite database for results
)
```

## Workflow

<div align="center">
    <img src="https://mermaid.ink/img/pako:eNp1k81uwjAMx1_FzWlIE7DLDkiAhIbWA9qkHTYOIXWrQZuUSVCH0N59zkc3wSBSYv__n2L7OA7CC2IQoSAPaIRdyJRvSQg3b5G_m8m1mT5UymErqVQjKCY0mJKV1I7UoXrblsJhwZLZssdE6SJVYgmUvq-KXm9U1g4fMn7RFLBKy-yqrThWjDVKVkDDN1l5NofC4QRKO_kVSMEe0fEwxApx-EmQbYGxQpqFtlnYWOZU4AZxP2l56Ybc4YUUbxLR8pzgH-EUQ4L5dH49qT6vr2ZXi1L4YvOqEG_PZn47qJGfSX1h6C04YS9TZbgmqmz51Jt1Yo8nN0c9Uu-EUdPO67A_6s0HdWjnXR6L6MdBzHxpkjEXkrRFPNZaqW7oy2X_KZJe6TNsNbTDH5-jFKWNTCnM-pHapEkBOx_sEF1Q5OgCtJl0rD-VZo-bsI22Pn5Fp3dVlF8I0znxhxDW7ZGfz4ZvHQ3h8PrPQRmTGWJI8FzAJsEOGp-QFmMxEo5JFh6nOr_J-P0Hwa1rNWmcKYe2yDoQfx9n63k" alt="OpenCell Workflow" />
</div>

## Architecture

OpenCell is structured around a modular pipeline design:

```
OpenCell/
├── src/
│   ├── core/
│   │   ├── transcriptomics/  # Single-cell RNA-seq analysis
│   │   └── utils/            # Shared utilities
│   └── workflow/
│       ├── database/         # Data persistence
│       ├── pipeline/         # Workflow orchestration
│       └── examples/         # Usage examples
├── alembic/                  # Database migrations
└── tests/                    # Test suite
```

## Performance Benchmarks

OpenCell has been optimized for performance, particularly when utilizing GPU acceleration:

| Dataset Size | CPU Runtime | GPU Runtime | Speedup |
|--------------|-------------|-------------|---------|
| 5K cells     | 10.5 min    | 3.2 min     | 3.3x    |
| 10K cells    | 24.8 min    | 6.5 min     | 3.8x    |
| 25K cells    | 68.3 min    | 15.1 min    | 4.5x    |

### Memory Usage Optimizations

We've implemented several memory optimization techniques:

- **Sparse matrix** operations where appropriate
- **Incremental processing** for large datasets
- **Efficient caching** of intermediate results
- **Memory-mapped files** for very large datasets

## Advanced Usage

### Custom Pathway Analysis

```python
# Run pathway analysis with specific organism and gene set
pathway_results = processor.pathway_analysis(
    n_top_genes=50,
    organism="human",  # Or "mouse"
)

# Access pathway results
for cluster, pathways in pathway_results["enrichr"].items():
    print(f"Cluster {cluster} enriched pathways:")
    for pathway in pathways["GO_Biological_Process_2021"][:5]:  # Top 5 GO terms
        print(f"  - {pathway['Term']} (p-adj={pathway['Adjusted P-value']:.1e})")
```

### Interactive Visualization Integration

```python
# Generate interactive plots (requires additional dependencies)
try:
    import scanpy.external as sce
    sce.exporting.cellbrowser(adata, "cellbrowser_dir", "OpenCell Analysis")
    print("Exported to Cell Browser format")
except ImportError:
    print("Cell Browser export requires additional dependencies")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

Nik Jois - [nikjois@llamasearch.ai](mailto:nikjois@llamasearch.ai)

Project Link: [https://github.com/llamasearchai/OpenCell](https://github.com/llamasearchai/OpenCell) 