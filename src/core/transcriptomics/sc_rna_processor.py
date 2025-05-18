import logging
import multiprocessing
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import yaml # Added for parameter saving/loading
import shutil # Moved to top

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import psutil
import threading

from ..utils.cache import cached_computation

# Optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False
    # logger.debug will be available after logger definition

# Optional GPU libraries
try:
    import cupy
    from cudf import DataFrame
    import cuml
    _cupy_cudf_cuml_available = True
except ImportError:
    _cupy_cudf_cuml_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

# Optional scRNA-seq analysis tools
try:
    import scrublet as scr
    _scrublet_available = True
except ImportError:
    _scrublet_available = False

try:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import r as rpy2_r 
    # import rpy2.robjects.numpy2ri as numpy2ri # If dense matrices are passed to R
    # Consider activating pandas2ri interface globally or locally before use
    # pandas2ri.activate() 
    _rpy2_available = True
except ImportError:
    _rpy2_available = False

try:
    import harmonypy
    _harmonypy_available = True
except ImportError:
    _harmonypy_available = False

try:
    from sklearn.metrics import silhouette_score
    _sklearn_silhouette_available = True
except ImportError:
    _sklearn_silhouette_available = False

# Optional pathway analysis tools
try:
    import gseapy as gp
    _gseapy_available = True
except ImportError:
    _gseapy_available = False

try:
    from goatools import obo_parser
    from goatools.go_enrichment import GOEnrichmentStudy
    import urllib.request
    _goatools_available = True
except ImportError:
    _goatools_available = False

# For SQLite integration
try:
    import sqlite_utils
    _sqlite_utils_available = True
except ImportError:
    _sqlite_utils_available = False

# Other utilities that were imported mid-function
from scipy.io import mmwrite, mmread 
from scipy.sparse import issparse 

logger = logging.getLogger(__name__)

# Now add debug logs for missing optional imports
if not _matplotlib_available:
    logger.debug("Matplotlib or Seaborn not found. Plotting functions will be limited.")
if not _cupy_cudf_cuml_available:
    logger.debug("CuPy, cuDF, or cuML not found. GPU acceleration for these specific libraries is unavailable.")
if not _torch_available:
    logger.debug("PyTorch not found. GPU availability checks and some GPU stats might be affected.")
if not _scrublet_available:
    logger.debug("Scrublet not found. Doublet detection with Scrublet will be skipped.")
if not _rpy2_available:
    logger.debug("rpy2 not found. R-dependent methods (DoubletFinder, scran, sctransform) will be skipped or fall back.")
if not _harmonypy_available:
    logger.debug("Harmonypy not found. Batch correction with Harmony will be skipped.")
if not _sklearn_silhouette_available:
    logger.debug("sklearn.metrics.silhouette_score not found. Silhouette score calculation will be skipped.")
if not _gseapy_available:
    logger.debug("gseapy not found. Pathway analysis with gseapy will be skipped.")
if not _goatools_available:
    logger.debug("goatools or urllib.request not found. GO enrichment with goatools will be skipped.")
if not _sqlite_utils_available:
    logger.debug("sqlite-utils not found. SQLite integration in save_results will be skipped.")


@dataclass
class ScRNAParameters:
    """Parameters for scRNA-seq analysis with enhanced options"""

    min_genes: int = 200
    min_cells: int = 3
    max_counts: int = 20000
    max_pct_mito: float = 20.0
    n_pcs: int = 50
    n_neighbors: int = 15
    resolution: float = 0.5
    use_doublet_detection: bool = True
    batch_correction: bool = False
    batch_key: Optional[str] = None
    normalization_method: str = "log1p"  # "log1p", "scran", "pearson_residuals", "sctransform"
    hvg_method: str = "seurat_v3"  # "seurat", "cell_ranger", "seurat_v3", "dispersion"
    n_hvgs: int = 2000
    clustering_method: str = "leiden"  # "leiden", "louvain"
    # Enhanced parameters
    n_jobs: int = max(1, multiprocessing.cpu_count() - 1)  # Parallel processing
    random_state: int = 42  # For reproducibility
    use_gpu: bool = False  # GPU acceleration where available
    save_intermediates: bool = False  # Save intermediate results
    marker_detection_method: str = "wilcoxon"  # "wilcoxon", "t-test", "logreg"
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    leiden_objective: str = "modularity"  # "modularity", "CPM" (Constant Potts Model)
    regress_out: List[str] = field(default_factory=lambda: ["total_counts", "pct_counts_mt"])
    use_harmony: bool = False
    cell_cycle_scoring: bool = False

    def save_to_yaml(self, filepath: Union[str, Path]):
        """Saves the parameters to a YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, sort_keys=False)
        logger.info(f"Parameters saved to {filepath}")

    @classmethod
    def load_from_yaml(cls, filepath: Union[str, Path]) -> 'ScRNAParameters':
        """Loads parameters from a YAML file."""
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Parameter file not found: {filepath}")
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
        with open(filepath, 'r') as f:
            params_dict = yaml.safe_load(f)
        
        # Ensure all fields are present, falling back to defaults if not
        # This handles cases where a new parameter is added to the class
        # and an old YAML file is loaded.
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_params = {k: v for k, v in params_dict.items() if k in valid_fields}
        
        return cls(**filtered_params)


class ScRNAProcessor:
    """
    Process and analyze single-cell RNA-seq data
    Enhanced with performance optimizations and additional methods
    """

    def __init__(self, parameters: Optional[ScRNAParameters] = None):
        """Enhanced initialization with:
        - Parameter validation
        - Environment checks
        - Resource monitoring
        """
        self.parameters = parameters or ScRNAParameters()
        self._validate_parameters()
        
        # GPU flags will be set by _check_gpu_system_readiness within _check_environment
        self.gpu_torch_cuda_available = False
        self.gpu_rapids_available = False
        self.actual_use_gpu = False

        self._check_environment() # This will call _check_gpu_system_readiness
        self._setup_resource_monitor() # Must be after _check_gpu_system_readiness sets actual_use_gpu
        
        self.adata = None
        self.results = {}
        self.intermediate_adatas = {}
        self.execution_times = {}
        self._setup_logging()

        # Configure scanpy settings based on parameters
        sc.settings.n_jobs = self.parameters.n_jobs
        sc.settings.verbosity = 2  # Slightly more verbose

        # Add benchmarking capability
        self.benchmarks = {
            'load_time': None,
            'preprocess_time': None,
            'pca_time': None,
            'clustering_time': None
        }

    def _setup_logging(self):
        """Configure logging for the processor"""
        self.logger = logging.getLogger(f"ScRNAProcessor.{id(self)}")
        self.logger.setLevel(logging.INFO)

    @cached_computation
    def load_data(self, data_path: Union[str, Path], format_type: str = "10x") -> ad.AnnData:
        """Enhanced data loading with validation"""
        try:
            if format_type == "10x":
                self.adata = sc.read_10x_mtx(data_path)
            elif format_type == "h5ad":
                self.adata = sc.read_h5ad(data_path)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
            # Validate loaded data
            if self.adata.n_obs == 0 or self.adata.n_vars == 0:
                raise ValueError("Empty dataset loaded")
                
            return self.adata
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def _enable_gpu_support(self):
        """Enable GPU acceleration for supported operations."""
        # This method is now effectively replaced by _check_gpu_system_readiness 
        # and direct checks in preprocess(). Keep it or remove it.
        # For now, let's comment out its old body if it wasn't removed by previous edits.
        # if _cupy_cudf_cuml_available:
        #     self._gpu_enabled = True 
        #     logger.info("CuPy, cuDF, and cuML found. GPU acceleration for them is possible if enabled by parameters.")
        # else:
        #     self._gpu_enabled = False
        #     logger.info("CuPy, cuDF, or cuML not found. GPU acceleration for them is unavailable.")
        pass # Marking as reviewed, effectively deprecated

    def preprocess(self, adata: Optional[ad.AnnData] = None) -> ad.AnnData:
        """Enhanced preprocessing with GPU support"""
        if adata is not None:
            self.adata = adata
        
        # if self._gpu_enabled: # Old check
        if self.actual_use_gpu and self.gpu_rapids_available:
            self.logger.info("GPU use requested and RAPIDS libraries available. Attempting GPU preprocessing.")
            return self._preprocess_gpu()
        else:
            if self.parameters.use_gpu and not self.gpu_rapids_available:
                self.logger.info("GPU use requested, but RAPIDS (CuPy/cuDF/cuML) libraries not found. Falling back to CPU preprocessing.")
            elif not self.parameters.use_gpu:
                self.logger.info("CPU preprocessing selected by parameters.")
            else: # actual_use_gpu is False but parameters.use_gpu was True (meaning no GPU system at all)
                self.logger.info("GPU use requested, but no suitable GPU system (Torch CUDA or RAPIDS) found. Falling back to CPU preprocessing.")
            return self._preprocess_cpu()

    def _preprocess_gpu(self) -> ad.AnnData:
        """Preprocess data using GPU acceleration (RAPIDS/CuPy path)."""
        # This path is taken if self.actual_use_gpu and self.gpu_rapids_available are True.
        # if not self._gpu_enabled: # This old flag is no longer the primary gatekeeper
        #     return self._preprocess_cpu()

        self.logger.info("Running RAPIDS/CuPy specific preprocessing steps.")
        # ... (Actual GPU code using cp, cudf, cuml would go here) ...
        self.logger.warning("GPU preprocessing path chosen (RAPIDS), but no specific GPU operations implemented here. Consider using GPU-accelerated libraries directly or ensuring Scanpy utilizes them. Falling back to CPU path as placeholder.")
        return self._preprocess_cpu() # Fallback to CPU path as placeholder

    def _score_cell_cycle(self):
        """Score cell cycle phases"""
        try:
            # Use well-established cell cycle genes
            s_genes = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL"]
            g2m_genes = [
                "HMGB2",
                "CDK1",
                "NUSAP1",
                "UBE2C",
                "BIRC5",
                "TPX2",
                "TOP2A",
                "NDC80",
                "CKS2",
                "NUF2",
                "CKS1B",
                "MKI67",
            ]

            # Convert gene names if necessary
            if self.adata.var_names[0].startswith("ENS"):
                logger.info("Gene IDs appear to be Ensembl IDs. Attempting to convert to gene symbols.")
                if "gene_symbols" in self.adata.var:
                    id_to_symbol = dict(zip(self.adata.var_names, self.adata.var["gene_symbols"], strict=False))
                    s_genes = [id_to_symbol.get(g, g) for g in s_genes]
                    g2m_genes = [id_to_symbol.get(g, g) for g in g2m_genes]

            logger.info("Scoring cell cycle phases")
            sc.tl.score_genes_cell_cycle(self.adata, s_genes=s_genes, g2m_genes=g2m_genes)

            logger.info(f"Cell cycle phases detected: {self.adata.obs['phase'].value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"Could not score cell cycle: {str(e)}")

    def _detect_doublets(self):
        """Detect potential doublets using Scrublet or DoubletFinder"""
        # try:
        #     import scrublet as scr
        if _scrublet_available:
            self.logger.info("Running doublet detection with Scrublet")
            scrub = scr.Scrublet(self.adata.X) # scr is globally available if _scrublet_available is True
            doublet_scores, predicted_doublets = scrub.scrub_doublets(
                min_counts=3, min_cells=3, min_gene_variability_pctl=85, n_prin_comps=30
            )

            self.adata.obs["doublet_score"] = doublet_scores
            self.adata.obs["predicted_doublet"] = predicted_doublets

            self.results["doublet_rate"] = float(np.mean(predicted_doublets))
            self.logger.info(f"Detected doublet rate: {self.results['doublet_rate']:.4f}")

            # Create violin plot of doublet scores
            # try:
            #     import matplotlib.pyplot as plt
            #     import seaborn as sns
            if _matplotlib_available:
                try:
                    plt.figure(figsize=(8, 6)) # plt, sns are globally available if _matplotlib_available is True
                    sns.violinplot(x=predicted_doublets, y=doublet_scores)
                    plt.xlabel("Predicted Doublet")
                    plt.ylabel("Doublet Score")
                    plt.title("Doublet Scores")
                    # Ensure figures directory exists
                    os.makedirs("figures", exist_ok=True)
                    plt.savefig("figures/doublet_scores.pdf")
                    plt.close()
                    self.results["doublet_figure_path"] = "figures/doublet_scores.pdf"
                except Exception as e:
                    self.logger.warning(f"Could not generate doublet figure: {str(e)}")
            else:
                self.logger.info("Matplotlib/Seaborn not available, skipping doublet score plot generation.")
            # except Exception as e:
            #     logger.warning(f"Could not generate doublet figure: {str(e)}")

        # except ImportError:
        #     logger.warning("Scrublet not installed. Trying alternative DoubletFinder approach...")
        else: # This else corresponds to `if _scrublet_available:`
            self.logger.warning("Scrublet not installed or not available. Trying alternative DoubletFinder approach...")
            # try:
            #     # Use R's DoubletFinder through rpy2 if available
            #     from rpy2.robjects import pandas2ri, r
            if _rpy2_available:
                try:
                    pandas2ri.activate() # Activate R interface
                    self.logger.info("Running doublet detection with DoubletFinder via R")

                    # Save temporary file for R
                    temp_file = "temp_for_doubletfinder.h5ad"
                    self.adata.write(temp_file)

                    # Run DoubletFinder in R
                    rpy2_r( # Use global rpy2_r
                        f"""
                    library(Seurat)
                    library(DoubletFinder)
                    library(SeuratDisk)
                    
                    seurat_obj <- LoadH5Seurat(\"{temp_file}\")
                    
                    seurat_obj <- NormalizeData(seurat_obj)
                    seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = \"vst\", nfeatures = 2000)
                    seurat_obj <- ScaleData(seurat_obj)
                    seurat_obj <- RunPCA(seurat_obj)
                    
                    sweep.res <- paramSweep_v3(seurat_obj, PCs = 1:20, sct = FALSE)
                    sweep.stats <- summarizeSweep(sweep.res, GT = FALSE)
                    bcmvn <- find.pK(sweep.stats)
                    optimal_pk <- as.numeric(as.character(bcmvn$pK[which.max(bcmvn$BCmetric)]))
                    
                    seurat_obj <- doubletFinder_v3(seurat_obj, 
                                                   PCs = 1:20, 
                                                   pN = 0.25, 
                                                   pK = optimal_pk, 
                                                   nExp = round(0.08 * ncol(seurat_obj)))
                    
                    doublet_column <- grep(\"DF.classifications\", colnames(seurat_obj@meta.data), value = TRUE)[1]
                    score_column <- grep(\"DF.scores\", colnames(seurat_obj@meta.data), value = TRUE)[1]
                    
                    doublet_results <- data.frame(
                      cell = rownames(seurat_obj@meta.data),
                      doublet_score = seurat_obj@meta.data[[score_column]],
                      predicted_doublet = seurat_obj@meta.data[[doublet_column]] == \"Doublet\"
                    )
                    
                    write.csv(doublet_results, \"doublet_results.csv\", row.names = FALSE)
                    
                    file.remove(\"{temp_file}\")
                    """
                    )

                    doublet_results_df = pd.read_csv("doublet_results.csv") # Read the cell column as a normal column first
                    # Find the cell identifier column used by Seurat (often rownames become a column or are implicit)
                    # Assuming the first column is the cell identifier if not explicitly named 'cell' by write.csv
                    # Or ensure R writes rownames explicitly to a 'cell' column that matches AnnData obs_names
                    # For safety, let's try to match with self.adata.obs_names
                    # This part needs careful handling of cell name matching between R and Python
                    
                    # A safer way to reintegrate doublet results:
                    # Ensure doublet_results_df is indexed by cell names that match adata.obs_names
                    # If R output CSV has cell names as first col and no header for it:
                    # doublet_results_df = pd.read_csv("doublet_results.csv", index_col=0)
                    # If R output CSV has a header like 'cell' for the cell names:
                    # doublet_results_df = pd.read_csv("doublet_results.csv").set_index('cell')
                    # The R script writes a 'cell' column. So set_index should work if column name is 'cell'.
                    try:
                        doublet_results_df = doublet_results_df.set_index('cell')
                        # Align with current adata.obs_names
                        self.adata.obs = self.adata.obs.join(doublet_results_df[['doublet_score', 'predicted_doublet']])
                        self.adata.obs['predicted_doublet'] = self.adata.obs['predicted_doublet'].fillna(False).astype(bool)

                        self.results["doublet_rate"] = float(np.mean(self.adata.obs["predicted_doublet"].dropna()))
                        self.logger.info(f"Detected doublet rate (DoubletFinder): {self.results['doublet_rate']:.4f}")
                    except Exception as e_join:
                        self.logger.error(f"Failed to join DoubletFinder results: {e_join}. Check cell name consistency.")

                    if os.path.exists("doublet_results.csv"): os.remove("doublet_results.csv")

                except Exception as e_rpy2:
                    self.logger.error(f"DoubletFinder via R failed: {e_rpy2}")
                    self.logger.warning("Skipping doublet detection as both Scrublet and DoubletFinder failed or are unavailable.")
            # except ImportError:
            #     logger.warning("Neither Scrublet nor rpy2 is installed. Skipping doublet detection.")
            else: # This else corresponds to `if _rpy2_available:`
                 self.logger.warning("Neither Scrublet nor rpy2 is installed/available. Skipping doublet detection.")

    def _normalize_data(self):
        """Normalize data with multiple available methods"""
        method = self.parameters.normalization_method
        self.logger.info(f"Normalizing data using method: {method}")

        if method == "log1p":
            # Standard scanpy normalization
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            self.logger.info("Completed log1p normalization")

        elif method == "scran":
            # Import scran normalization via external R process
            # try:
            #     from rpy2.robjects import pandas2ri, r
            if _rpy2_available:
                try:
                    pandas2ri.activate()
                    # Execute scran normalization in R
                    self.logger.info("Running scran normalization via R")

                    # Save temporary files for R
                    temp_dir = f"temp_scran_{time.time()}"
                    os.makedirs(temp_dir, exist_ok=True)
                    counts_path = f"{temp_dir}/counts.mtx"
                    cell_path = f"{temp_dir}/cells.csv"

                    # Save matrix and metadata
                    # from scipy.io import mmwrite # Use global
                    # from scipy.sparse import issparse # Use global

                    if issparse(self.adata.X):
                        mmwrite(counts_path, self.adata.X)
                    else:
                        mmwrite(counts_path, scipy.sparse.csr_matrix(self.adata.X))

                    self.adata.obs.to_csv(cell_path)

                    # Define R script for scran normalization
                    rpy2_r( # Use global rpy2_r
                        '''
                    normalize_scran <- function(counts_path, cell_path) {
                        library(scran)
                        library(Matrix)
                        
                        # Read data
                        counts <- readMM(counts_path)
                        cells <- read.csv(cell_path, row.names=1)
                        
                        # Create SingleCellExperiment
                        library(SingleCellExperiment)
                        sce <- SingleCellExperiment(list(counts=counts), colData=cells)
                        
                        # Run normalization
                        set.seed(1234)
                        clusters <- quickCluster(sce)
                        sce <- computeSumFactors(sce, clusters=clusters)
                        sce <- logNormCounts(sce)
                        
                        # Return normalized matrix
                        norm_factors <- sizeFactors(sce)
                        write.csv(data.frame(size_factor=norm_factors), "size_factors.csv")
                        
                        # Return log-normalized data
                        normalized <- logcounts(sce)
                        writeMM(normalized, "normalized.mtx")
                        
                        return(invisible(NULL))
                    }
                    
                    # Run normalization
                    normalize_scran("'''
                        + counts_path
                        + '''", "'''
                        + cell_path
                        + """")
                    """
                    )

                    # Load results back
                    size_factors = pd.read_csv("size_factors.csv", index_col=0)
                    self.adata.obs["size_factors"] = size_factors["size_factor"].values

                    # Load normalized matrix
                    # from scipy.io import mmread # Use global
                    norm_mat = mmread("normalized.mtx")
                    self.adata.X = norm_mat

                    # Clean up
                    os.remove("size_factors.csv")
                    os.remove("normalized.mtx")
                    # import shutil # Use global
                    shutil.rmtree(temp_dir)
                    self.logger.info("Completed scran normalization")

                except Exception as e_r_scran: # Catch R execution errors or other issues
                    self.logger.error(f"Scran normalization via R failed: {e_r_scran}")
                    self.logger.warning("Falling back to log1p normalization.")
                    sc.pp.normalize_total(self.adata, target_sum=1e4)
                    sc.pp.log1p(self.adata)
            else:
            # except ImportError:
                self.logger.warning("rpy2 not available, falling back to log1p normalization for scran method.")
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)

        elif method == "pearson_residuals":
            # Implement Pearson residuals normalization (better for UMI data)
            # from scipy.sparse import issparse # Use global
            self.logger.info("Calculating Pearson residuals")

            if issparse(self.adata.X):
                X = self.adata.X.toarray()
            else:
                X = self.adata.X.copy()

            # Calculate mean and variance for each gene
            mean_expr = np.mean(X, axis=0)
            var_expr = np.var(X, axis=0)

            # Calculate Pearson residuals with improved formula
            # Using regularization factor to avoid division by zero
            # Ensure mean_expr and var_expr are not zero to avoid division by zero or sqrt of negative
            # Add a small epsilon to denominators
            epsilon = 1e-8
            residuals = (X - mean_expr) / np.sqrt(var_expr + epsilon) 
            # Original scanpy pearson_residuals uses X_sparse.mean(axis=0).A.squeeze() for mu
            # and X_csc.power(2).mean(axis=0).A.squeeze() - mu ** 2 for variance
            # The direct np.var might differ slightly for sparse data if not handled carefully.
            # For simplicity, sticking to np.mean and np.var on dense array for now.
            # A more robust implementation might use scanpy's internal logic if available as a utility.
            # sc.experimental.pp.highly_variable_genes(self.adata, flavor="pearson_residuals", n_top_genes=self.adata.n_vars) might calculate them.
            # self.adata.X = sc.experimental.pp.normalize_pearson_residuals(self.adata, inplace=False) # if using experimental
            
            # Cap values to avoid extreme residuals (Scanpy does this too)
            residuals = np.clip(residuals, a_min=None, a_max=np.sqrt(self.adata.n_obs))

            # Store original data if not already present
            if "counts" not in self.adata.layers:
                 self.adata.layers["counts"] = self.adata.X.copy() # Save before overwriting X

            # Set residuals as main data matrix
            self.adata.X = residuals
            self.logger.info("Completed Pearson residuals normalization")

        elif method == "sctransform":
            # Use sctransform approach via R
            # try:
            #     from rpy2.robjects import pandas2ri, r
            if _rpy2_available:
                try:
                    pandas2ri.activate()
                    self.logger.info("Running sctransform normalization via R")

                    # Save data to temp files
                    temp_file = f"temp_sctransform_{time.time()}.h5ad"
                    self.adata.write(temp_file)

                    # Run sctransform in R
                    rpy2_r( # Use global rpy2_r
                        f"""
                    library(Seurat)
                    library(sctransform)
                    library(SeuratDisk)
                    
                    Convert(\"{temp_file}\", dest = \"h5seurat\", overwrite = TRUE)
                    seurat_obj <- LoadH5Seurat(\"{temp_file}.h5seurat\")
                    
                    seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
                    
                    sct_data <- GetAssayData(seurat_obj, slot = \"data\", assay = \"SCT\")
                    writeMM(sct_data, \"sct_normalized.mtx\")
                    
                    writeLines(rownames(sct_data), \"gene_names.txt\")
                    writeLines(colnames(sct_data), \"cell_names.txt\")
                    
                    file.remove(\"{temp_file}\")
                    file.remove(\"{temp_file}.h5seurat\")
                    """
                    )

                    # Load results back
                    # from scipy.io import mmread # Use global
                    sct_mat = mmread("sct_normalized.mtx")

                    with open("gene_names.txt", "r") as f:
                        gene_names = [line.strip() for line in f]
                    with open("cell_names.txt", "r") as f:
                        cell_names = [line.strip() for line in f]

                    sct_adata = ad.AnnData(
                        X=sct_mat.T, obs=self.adata.obs.loc[cell_names], var=pd.DataFrame(index=gene_names)
                    )

                    if "counts" not in self.adata.layers:
                        self.adata.layers["counts"] = self.adata.X.copy()
                    self.adata.X = sct_adata.X

                    os.remove("sct_normalized.mtx")
                    os.remove("gene_names.txt")
                    os.remove("cell_names.txt")
                    self.logger.info("Completed sctransform normalization")

                except Exception as e_r_sct:
                    self.logger.error(f"SCTransform normalization via R failed: {e_r_sct}")
                    self.logger.warning("Falling back to log1p normalization.")
                    sc.pp.normalize_total(self.adata, target_sum=1e4)
                    sc.pp.log1p(self.adata)
            else:
            # except ImportError:
                self.logger.warning("rpy2 not available, falling back to log1p normalization for sctransform method.")
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        # Save normalized state
        if self.parameters.save_intermediates:
            self.intermediate_adatas["normalized"] = self.adata.copy()

    def _find_variable_genes(self):
        """Identify highly variable genes with optimized methods"""
        method = self.parameters.hvg_method
        n_hvgs = self.parameters.n_hvgs

        logger.info(f"Finding {n_hvgs} highly variable genes using method: {method}")

        # Check for batch correction mode
        batch_key = self.parameters.batch_key if self.parameters.batch_correction else None
        if batch_key and batch_key not in self.adata.obs:
            logger.warning(f"Batch key '{batch_key}' not found in data. Reverting to non-batch mode.")
            batch_key = None

        if method == "seurat":
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs, flavor="seurat", batch_key=batch_key)
        elif method == "cell_ranger":
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs, flavor="cell_ranger", batch_key=batch_key)
        elif method == "seurat_v3":
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs, flavor="seurat_v3", batch_key=batch_key)
        elif method == "dispersion":
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_hvgs,
                flavor="seurat_v3",  # This is most similar to dispersion approach
                batch_key=batch_key,
            )
        else:
            logger.warning(f"Unknown HVG method: {method}. Using seurat_v3 instead.")
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_hvgs, flavor="seurat_v3", batch_key=batch_key)

        # Filter to highly variable genes for downstream analysis
        self.adata = self.adata[:, self.adata.var.highly_variable]
        logger.info(f"Selected {self.adata.n_vars} highly variable genes")

        # Save HVG state
        if self.parameters.save_intermediates:
            self.intermediate_adatas["hvg_selected"] = self.adata.copy()

    def _regress_out_technical_factors(self):
        """Regress out technical factors from data"""
        factors = self.parameters.regress_out

        if not factors:
            return

        # Ensure all factors exist in the data
        valid_factors = [f for f in factors if f in self.adata.obs.columns]
        if len(valid_factors) < len(factors):
            missing = set(factors) - set(valid_factors)
            logger.warning(f"Could not find these factors for regression: {missing}")

        if not valid_factors:
            logger.warning("No valid factors for regression. Skipping.")
            return

        logger.info(f"Regressing out technical factors: {valid_factors}")
        sc.pp.regress_out(self.adata, valid_factors)
        logger.info("Completed regression")

    def run_pca(self) -> None:
        """
        Run PCA dimensionality reduction with enhanced options
        """
        start_time = time.time()

        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() and preprocess() first.")

        logger.info(f"Running PCA with {self.parameters.n_pcs} components")

        # Check if data is too large for randomized PCA
        use_randomized = self.adata.n_vars > 10000 or self.adata.n_obs > 10000
        svd_solver = "randomized" if use_randomized else "arpack"

        sc.tl.pca(
            self.adata,
            svd_solver=svd_solver,
            n_comps=self.parameters.n_pcs,
            random_state=self.parameters.random_state,
            use_highly_variable=True,
        )

        # Store variance explained for reporting
        self.results["variance_explained"] = self.adata.uns["pca"]["variance_ratio"].tolist()
        self.results["pca_variance_ratio"] = float(np.sum(self.adata.uns["pca"]["variance_ratio"]))

        # Create elbow plot for PCA
        if _matplotlib_available:
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(np.cumsum(self.adata.uns["pca"]["variance_ratio"]), "o-")
                plt.axhline(y=0.9, linestyle="--", color="red")
                plt.xlabel("Number of PCs")
                plt.ylabel("Cumulative variance explained")
                plt.title("PCA Elbow Plot")
                os.makedirs("figures", exist_ok=True) # Ensure figures directory exists
                plt.savefig("figures/pca_elbow.pdf")
                plt.close()
                self.results["pca_elbow_path"] = "figures/pca_elbow.pdf"
            except Exception as e:
                self.logger.warning(f"Could not generate PCA elbow plot: {str(e)}")
        else:
            self.logger.info("Matplotlib not available, skipping PCA elbow plot generation.")

        elapsed = time.time() - start_time
        self.execution_times["run_pca"] = elapsed
        logger.info(
            f"PCA complete in {elapsed:.2f}s. Top PC variance explained: {self.results['variance_explained'][0]:.4f}"
        )

    def batch_correction(self, batch_key: str = None) -> None:
        """
        Perform batch correction with expanded options

        Args:
            batch_key: Column in adata.obs specifying batch information
        """
        start_time = time.time()

        if not self.parameters.batch_correction:
            logger.info("Batch correction skipped")
            return

        batch_key = batch_key or self.parameters.batch_key
        if batch_key is None or batch_key not in self.adata.obs:
            logger.warning(f"Batch key {batch_key} not found in data. Skipping batch correction.")
            return

        # Save pre-batch correction state
        if self.parameters.save_intermediates:
            self.intermediate_adatas["pre_batch_correction"] = self.adata.copy()

        logger.info(f"Performing batch correction using {batch_key}")

        # If harmony is selected
        if self.parameters.use_harmony:
            # try:
            #     # Use harmony for batch correction
            #     import harmonypy
            if _harmonypy_available:
                try:
                    self.logger.info("Running Harmony batch correction")
                    # Ensure batch_key exists and is suitable for harmonypy
                    if batch_key not in self.adata.obs:
                        self.logger.warning(f"Batch key '{batch_key}' not found for Harmony. Falling back to Combat or no correction.")
                        # Decide fallback: Combat or nothing. For now, let Combat try.
                    else: 
                        harmony_out = harmonypy.run_harmony(
                            self.adata.obsm["X_pca"], self.adata.obs, batch_key, max_iter_harmony=50
                        )
                        self.adata.obsm["X_pca_harmony"] = harmony_out.Z_corr
                        self.logger.info("Using harmony-corrected PCs for downstream analysis")
                        elapsed = time.time() - start_time
                        self.execution_times["batch_correction_harmony"] = elapsed # Specific timing
                        self.logger.info(f"Harmony batch correction complete in {elapsed:.2f}s")
                        return # Harmony done, exit batch_correction method
                except Exception as e_harmony:
                    self.logger.error(f"Error during Harmony batch correction: {e_harmony}. Falling back to Combat.")
            else:
            # except ImportError:
                self.logger.warning("Harmony (harmonypy) not installed/available. Falling back to Combat for batch correction if use_harmony was true.")

        # Combat batch correction as fallback (if Harmony not used, failed, or not available)
        # This section will be reached if self.parameters.use_harmony is False, 
        # or if it was True but harmonypy was not available or failed.
        self.logger.info("Attempting Combat batch correction (either as primary or fallback).")
        try:
            # import pandas as pd # pandas (pd) is globally imported

            if not pd.api.types.is_categorical_dtype(self.adata.obs[batch_key]):
                self.adata.obs[batch_key] = self.adata.obs[batch_key].astype("category")

            self.logger.info(f"Running Combat batch correction on key: {batch_key}")
            sc.pp.combat(self.adata, key=batch_key)

            self.logger.info("Rerunning PCA on Combat batch-corrected data")
            sc.tl.pca(
                self.adata,
                svd_solver="arpack", # Consider making svd_solver consistent with main run_pca
                n_comps=self.parameters.n_pcs,
                random_state=self.parameters.random_state,
            )

            elapsed = time.time() - start_time # This captures time for combat + pca
            self.execution_times["batch_correction_combat"] = elapsed
            self.logger.info(f"Combat batch correction and re-PCA complete in {elapsed:.2f}s")

        except Exception as e_combat:
            self.logger.error(f"Error during Combat batch correction: {str(e_combat)}")
            self.logger.info("Proceeding without batch correction after Combat failure.")
            # Restore pre-batch PCA if combat fails? Or trust that adata is not too messed up?
            # For now, we just log and continue.
            # If pre_batch_correction intermediate was saved, could potentially revert.

    def run_clustering(self) -> None:
        """
        Run neighbor graph construction and clustering with enhanced options
        """
        start_time = time.time()

        if self.adata is None or "X_pca" not in self.adata.obsm:
            raise ValueError("PCA not computed. Call run_pca() first.")

        logger.info(f"Computing neighbor graph with {self.parameters.n_neighbors} neighbors")

        # Check if we should use harmony-corrected PCs
        use_rep = "X_pca_harmony" if "X_pca_harmony" in self.adata.obsm else "X_pca"

        # Enhanced neighbor graph construction
        sc.pp.neighbors(
            self.adata,
            n_neighbors=self.parameters.n_neighbors,
            n_pcs=self.parameters.n_pcs,
            use_rep=use_rep,
            random_state=self.parameters.random_state,
            metric="cosine",  # Better than euclidean for high-dimensional data
        )

        logger.info(
            f"Running {self.parameters.clustering_method} clustering with resolution {self.parameters.resolution}"
        )

        if self.parameters.clustering_method == "leiden":
            sc.tl.leiden(
                self.adata,
                resolution=self.parameters.resolution,
                random_state=self.parameters.random_state,
                objective_function=self.parameters.leiden_objective,
            )
            cluster_key = "leiden"
        elif self.parameters.clustering_method == "louvain":
            sc.tl.louvain(self.adata, resolution=self.parameters.resolution, random_state=self.parameters.random_state)
            cluster_key = "louvain"
        else:
            raise ValueError(f"Unsupported clustering method: {self.parameters.clustering_method}")

        # Count number of cells in each cluster
        cluster_counts = self.adata.obs[cluster_key].value_counts().to_dict()
        self.results["cluster_counts"] = cluster_counts
        self.results["n_clusters"] = len(cluster_counts)

        # Run silhouette analysis to evaluate clustering
        if _sklearn_silhouette_available:
            try:
                # Calculate silhouette score on PCA space (or harmony corrected if used)
                sil_score = silhouette_score(
                    self.adata.obsm[use_rep], self.adata.obs[cluster_key].values, random_state=self.parameters.random_state
                )
                self.results["silhouette_score"] = float(sil_score)
                self.logger.info(f"Clustering silhouette score: {sil_score:.4f}")
            except Exception as e:
                self.logger.warning(f"Could not calculate silhouette score: {str(e)}")
        else:
            self.logger.info("sklearn.metrics.silhouette_score not available, skipping silhouette score calculation.")

        elapsed = time.time() - start_time
        self.execution_times["run_clustering"] = elapsed
        logger.info(f"Clustering complete in {elapsed:.2f}s. Identified {self.results['n_clusters']} clusters")

    def run_umap(self) -> None:
        """
        Run UMAP dimensionality reduction for visualization with enhanced options
        """
        start_time = time.time()

        if self.adata is None or "neighbors" not in self.adata.uns:
            raise ValueError("Neighbor graph not computed. Call run_clustering() first.")

        logger.info("Running UMAP with enhanced parameters")

        # Run UMAP with custom parameters
        sc.tl.umap(
            self.adata,
            min_dist=self.parameters.umap_min_dist,
            spread=self.parameters.umap_spread,
            random_state=self.parameters.random_state,
        )

        # Save coordinates for visualization
        self.results["umap_coordinates"] = {
            "x": self.adata.obsm["X_umap"][:, 0].tolist(),
            "y": self.adata.obsm["X_umap"][:, 1].tolist(),
        }

        # Run t-SNE as an alternative visualization
        try:
            logger.info("Running t-SNE as alternative visualization")
            sc.tl.tsne(
                self.adata, use_rep="X_pca", random_state=self.parameters.random_state, n_jobs=self.parameters.n_jobs
            )

            # Save t-SNE coordinates
            self.results["tsne_coordinates"] = {
                "x": self.adata.obsm["X_tsne"][:, 0].tolist(),
                "y": self.adata.obsm["X_tsne"][:, 1].tolist(),
            }
        except Exception as e:
            logger.warning(f"Could not compute t-SNE: {str(e)}")

        elapsed = time.time() - start_time
        self.execution_times["run_umap"] = elapsed
        logger.info(f"Dimensionality reduction complete in {elapsed:.2f}s")

    def find_markers(self, groupby: str = None, method: str = None) -> pd.DataFrame:
        """
        Find marker genes for clusters with multiple methods

        Args:
            groupby: Column in adata.obs for grouping cells, default is clustering result
            method: Method for differential expression, default from parameters

        Returns:
            DataFrame with marker genes
        """
        start_time = time.time()

        if self.adata is None:
            raise ValueError("No data loaded. Process the data first.")

        if groupby is None:
            if "leiden" in self.adata.obs:
                groupby = "leiden"
            elif "louvain" in self.adata.obs:
                groupby = "louvain"
            else:
                raise ValueError("No clustering results found. Run clustering first.")

        method = method or self.parameters.marker_detection_method

        # Check if the method is valid
        valid_methods = ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"]
        if method not in valid_methods:
            logger.warning(f"Method {method} not recognized. Using wilcoxon instead.")
            method = "wilcoxon"

        logger.info(f"Finding marker genes for {groupby} groups using {method} method")

        # Run differential expression with selected method
        sc.tl.rank_genes_groups(
            self.adata,
            groupby=groupby,
            method=method,
            pts=True,  # Calculate percentage of cells expressing genes
            key_added=f"rank_genes_{method}",
        )

        # Create a DataFrame with marker genes, one per cluster
        markers_df = sc.get.rank_genes_groups_df(self.adata, group=None, key=f"rank_genes_{method}")

        # Enhance markers DataFrame with additional information
        markers_df["group"] = markers_df["group"].astype(str)  # Ensure group is string

        # Add average log fold change
        if "logfoldchanges" in markers_df.columns:
            markers_df["avg_logFC"] = markers_df["logfoldchanges"]

        # Add percentage of cells expressing the gene in the cluster
        if "pts" in markers_df.columns:
            pct_cols = [col for col in markers_df.columns if col.startswith("pts_")]
            for pct_col in pct_cols:
                group = pct_col.replace("pts_", "")
                markers_df.loc[markers_df["group"] == group, "pct_expressing"] = markers_df[pct_col]

        # Add mean expression within each group
        try:
            for group in markers_df["group"].unique():
                group_cells = self.adata.obs[groupby] == group
                group_genes = markers_df.loc[markers_df["group"] == group, "names"].values
                for gene in group_genes:
                    if gene in self.adata.var_names:
                        gene_idx = self.adata.var_names.get_loc(gene)
                        mean_expr = np.mean(self.adata.X[group_cells, gene_idx])
                        markers_df.loc[
                            (markers_df["group"] == group) & (markers_df["names"] == gene), "mean_expression"
                        ] = mean_expr
        except Exception as e:
            logger.warning(f"Could not calculate mean expression for marker genes: {str(e)}")

        # Store results
        self.results["markers"] = markers_df.to_dict("records")

        # Create marker heatmap
        try:
            sc.pl.rank_genes_groups_heatmap(
                self.adata,
                n_genes=10,
                groupby=groupby,
                key=f"rank_genes_{method}",
                show=False,
                save="markers_heatmap.pdf",
            )
            self.results["marker_heatmap_path"] = "figures/markers_heatmap.pdf"
        except Exception as e:
            logger.warning(f"Could not generate marker heatmap: {str(e)}")

        # Create marker dotplot
        try:
            sc.pl.rank_genes_groups_dotplot(
                self.adata,
                n_genes=5,
                groupby=groupby,
                key=f"rank_genes_{method}",
                show=False,
                save="markers_dotplot.pdf",
            )
            self.results["marker_dotplot_path"] = "figures/markers_dotplot.pdf"
        except Exception as e:
            logger.warning(f"Could not generate marker dotplot: {str(e)}")

        elapsed = time.time() - start_time
        self.execution_times["find_markers"] = elapsed
        logger.info(f"Found {len(markers_df)} marker genes in {elapsed:.2f}s")

        return markers_df

    def trajectory_inference(self, method: str = "paga", start_cluster: str = None) -> None:
        """
        Perform trajectory inference analysis with multiple methods

        Args:
            method: Method for trajectory inference ('paga', 'dpt', 'velocity')
            start_cluster: Starting cluster for trajectory, auto-detected if None
        """
        start_time = time.time()

        if self.adata is None or "neighbors" not in self.adata.uns:
            raise ValueError("Neighbor graph not computed. Run clustering first.")

        logger.info(f"Running trajectory inference using {method}")

        # Determine clustering key
        cluster_key = "leiden" if "leiden" in self.adata.obs else "louvain"

        if method == "paga":
            # PAGA for trajectory analysis with enhanced options
            sc.tl.paga(
                self.adata,
                groups=cluster_key,
                model="v1.2",  # Latest model with better connectivity estimation
                use_rna_velocity=False,
            )

            # Store PAGA connectivity for visualization
            self.results["paga_connectivities"] = self.adata.uns["paga"]["connectivities"].toarray().tolist()

            # Generate PAGA plot
            try:
                sc.pl.paga(
                    self.adata,
                    threshold=0.05,
                    layout="fr",  # Force-directed layout
                    show=False,
                    save="paga_trajectory.pdf",
                )
                self.results["paga_trajectory_path"] = "figures/paga_trajectory.pdf"
            except Exception as e:
                logger.warning(f"Could not generate PAGA plot: {str(e)}")

            # Initialize UMAP with PAGA for better visualization
            sc.tl.umap(self.adata, init_pos="paga")

        elif method == "dpt":
            # Diffusion pseudotime analysis
            sc.tl.diffmap(self.adata, n_comps=15)

            # Determine root cell/cluster
            if start_cluster is None:
                # Try to identify root cluster using marker genes or cell cycle scores
                if "phase" in self.adata.obs:
                    # Use G1 phase cells as root
                    g1_clusters = self.adata.obs[self.adata.obs["phase"] == "G1"][cluster_key].value_counts()
                    if not g1_clusters.empty:
                        start_cluster = g1_clusters.idxmax()
                        logger.info(f"Auto-selected start cluster {start_cluster} based on cell cycle phase")

                # If still None, use first cluster
                if start_cluster is None:
                    start_cluster = self.adata.obs[cluster_key].cat.categories[0]
                    logger.info(f"Using first cluster {start_cluster} as root")

            # Get a cell from the root cluster
            root_cell = np.where(self.adata.obs[cluster_key] == start_cluster)[0][0]

            # Run diffusion pseudotime
            sc.tl.dpt(
                self.adata,
                n_branchings=3,  # Allow multiple branching
                n_dcs=10,  # Use 10 diffusion components
                root=root_cell,
            )

            # Store pseudotime values
            self.results["pseudotime_values"] = self.adata.obs["dpt_pseudotime"].tolist()
            self.results["start_cluster"] = start_cluster

            # Create pseudotime plot
            try:
                sc.pl.draw_graph(
                    self.adata, color=["dpt_pseudotime", cluster_key], show=False, save="diffusion_pseudotime.pdf"
                )
                self.results["dpt_trajectory_path"] = "figures/diffusion_pseudotime.pdf"
            except Exception as e:
                logger.warning(f"Could not generate DPT plot: {str(e)}")

        elif method == "velocity":
            # RNA Velocity requires additional data
            logger.warning(
                "RNA Velocity requires spliced/unspliced counts. Please use velocyto or kallisto for preprocessing."
            )
            logger.warning("Falling back to PAGA trajectory inference")
            self.trajectory_inference(method="paga", start_cluster=start_cluster)
            return

        else:
            logger.warning(f"Unknown trajectory method: {method}. Falling back to PAGA.")
            self.trajectory_inference(method="paga", start_cluster=start_cluster)
            return

        elapsed = time.time() - start_time
        self.execution_times["trajectory_inference"] = elapsed
        logger.info(f"Trajectory inference complete in {elapsed:.2f}s")

    def pathway_analysis(
        self, markers_df: pd.DataFrame = None, n_top_genes: int = 50, organism: str = "human"
    ) -> Dict:
        """
        Perform pathway enrichment analysis on marker genes

        Args:
            markers_df: DataFrame of marker genes, if None uses result from find_markers
            n_top_genes: Number of top marker genes to use per cluster
            organism: Organism for gene sets ('human' or 'mouse')

        Returns:
            Dictionary with pathway analysis results
        """
        start_time = time.time()

        if markers_df is None and "markers" not in self.results:
            # Run marker detection first
            markers_df = self.find_markers()
        elif markers_df is None:
            # Convert stored markers back to DataFrame
            markers_df = pd.DataFrame(self.results["markers"])

        # Try different pathway analysis methods
        results = {}

        # try:
        #     # First attempt gseapy if available
        #     import gseapy as gp
        if _gseapy_available:
            try:
                self.logger.info(f"Running pathway analysis with gseapy for {organism}")
                gene_lists = {}
                for group in markers_df["group"].unique():
                    group_markers = markers_df[markers_df["group"] == group].sort_values("scores", ascending=False)
                    gene_lists[f"Cluster_{group}"] = group_markers["names"].head(n_top_genes).tolist()

                if organism.lower() == "human":
                    gene_sets = ["GO_Biological_Process_2021", "KEGG_2021_Human"]
                elif organism.lower() == "mouse":
                    gene_sets = ["GO_Biological_Process_2021", "KEGG_2021_Mouse"]
                else:
                    self.logger.warning(f"Unsupported organism '{organism}' for gseapy. Falling back or skipping gseapy.")
                    # raise ValueError(f"Unsupported organism: {organism}. Use 'human' or 'mouse'.")
                    # To avoid stopping the whole pipeline, log warning and gseapy_success = False
                    # This means results["enrichr"] might not be populated.
                    pass # Allow to proceed to goatools if gseapy organism is bad
                
                if gene_sets: # Proceed only if gene_sets were defined
                    enrichment_results = {}
                    for cluster_name, gene_list in gene_lists.items():
                        if not gene_list:
                            self.logger.info(f"No marker genes for {cluster_name}, skipping gseapy enrichment.")
                            continue
                        enr = gp.enrichr(
                            gene_list=gene_list, gene_sets=gene_sets, organism=organism, outdir=None, cutoff=0.05
                        )
                        cluster_results = {}
                        if enr is not None and hasattr(enr, 'results') and enr.results:
                            for gene_set_db in gene_sets:
                                if gene_set_db in enr.results:
                                    df = enr.results[gene_set_db]
                                    if not df.empty:
                                        df = df[df["Adjusted P-value"] < 0.05]
                                        if not df.empty:
                                            cluster_results[gene_set_db] = df.head(10).to_dict("records")
                        enrichment_results[cluster_name] = cluster_results
                    results["enrichr"] = enrichment_results
                self.logger.info("Completed enrichr pathway analysis with gseapy.")
            except Exception as e_gseapy:
                self.logger.error(f"gseapy pathway analysis failed: {e_gseapy}")
        else:
        # except ImportError:
            self.logger.warning("gseapy not available. Trying alternative method or skipping.")

        # try:
        #     # Try with goatools
        #     from goatools import obo_parser
        #     from goatools.go_enrichment import GOEnrichmentStudy
        if _goatools_available:
            try:
                self.logger.info("Running GO enrichment with goatools")
                obo_file = "go-basic.obo"
                if not os.path.exists(obo_file):
                    # import urllib.request # Use global
                    self.logger.info("Downloading GO OBO file")
                    urllib.request.urlretrieve("http://purl.obolibrary.org/obo/go/go-basic.obo", obo_file)

                go_dag = obo_parser.GODag(obo_file)
                background_genes = self.adata.var_names.tolist()
                
                # Check if adata.var_names has gene symbols appropriate for GO analysis
                # This is a placeholder for actual gene ID type checking / conversion if needed
                if not all(isinstance(gene, str) for gene in background_genes):
                    self.logger.warning("Background genes for goatools are not all strings. GO analysis might be unreliable.")

                gene_lists_goatools = {}
                for group in markers_df["group"].unique():
                    group_markers = markers_df[markers_df["group"] == group].sort_values("scores", ascending=False)
                    # Ensure marker genes are also strings
                    gene_lists_goatools[f"Cluster_{group}"] = [str(g) for g in group_markers["names"].head(n_top_genes).tolist() if g]
                
                enrichment_results_goatools = {}
                for cluster_name, gene_list in gene_lists_goatools.items():
                    if not gene_list:
                        self.logger.info(f"No marker genes for {cluster_name}, skipping goatools enrichment.")
                        continue
                    goea_study = GOEnrichmentStudy(
                        background_genes, 
                        {}, # gene2go mapping; goatools can create this if associations are provided or from an annotation file
                        go_dag, 
                        propagate_counts=True, 
                        alpha=0.05, 
                        methods=["fdr_bh"]
                    )
                    goea_results = goea_study.run_study(gene_list)
                    sig_results = [r for r in goea_results if r.p_fdr_bh < 0.05]
                    if sig_results:
                        enrichment_results_goatools[cluster_name] = [
                            {
                                "GO_ID": r.GO,
                                "Term": go_dag[r.GO].name,
                                "Adjusted P-value": r.p_fdr_bh,
                                "Genes": [str(item) for item in r.study_items], # Ensure genes are strings
                            }
                            for r in sig_results[:10]
                        ]
                results["goatools"] = enrichment_results_goatools
                self.logger.info("Completed GO enrichment analysis with goatools.")
            except Exception as e_goatools:
                self.logger.error(f"goatools pathway analysis failed: {e_goatools}")
        else:
        # except ImportError:
            self.logger.warning("goatools not available. Skipping GO pathway analysis.")

        self.results["pathway_analysis"] = results

        elapsed = time.time() - start_time
        self.execution_times["pathway_analysis"] = elapsed
        logger.info(f"Pathway analysis complete in {elapsed:.2f}s")

        return results

    def run_analysis(self, adata: Optional[ad.AnnData] = None) -> ad.AnnData:
        """Enhanced analysis pipeline with benchmarking"""
        with self._benchmark('total_analysis'):
            if adata is not None:
                self.adata = adata
                
            # Track memory before each major step
            self._track_memory('pre_preprocess')
            
            # Enhanced preprocessing
            with self._benchmark('preprocess'):
                self.preprocess()
                
            self._track_memory('post_preprocess')
            
            # PCA with improved logging
            with self._benchmark('pca'):
                self.run_pca()
                
            # Run batch correction if enabled
            self.batch_correction()
            
            # Run clustering
            with self._benchmark('clustering'):
                self.run_clustering()
                
            # Run UMAP for visualization
            with self._benchmark('umap'):
                self.run_umap()
                
            # Find marker genes
            with self._benchmark('find_markers'):
                self.find_markers()
                
            # Run trajectory inference if the data has enough cells
            if self.adata.n_obs >= 500:
                with self._benchmark('trajectory_inference'):
                    self.trajectory_inference()
                
            # Run pathway analysis
            with self._benchmark('pathway_analysis'):
                self.pathway_analysis()
            
        return self.adata
        
    def _benchmark(self, name: str):
        """Context manager for timing operations"""
        return BenchmarkContext(self, name)
        
    def _track_memory(self, stage: str):
        """Record memory usage at specific stages"""
        process = psutil.Process(os.getpid())
        self.results[f'memory_{stage}'] = process.memory_info().rss / (1024 ** 2)

    def _validate_parameters(self):
        """Precise parameter validation"""
        if not isinstance(self.parameters, ScRNAParameters):
            raise TypeError("parameters must be ScRNAParameters instance")
            
        if self.parameters.n_pcs < 5 or self.parameters.n_pcs > 100:
            raise ValueError("n_pcs must be between 5 and 100")
            
        if self.parameters.n_neighbors < 5 or self.parameters.n_neighbors > 100:
            raise ValueError("n_neighbors must be between 5 and 100")
            
        # Additional validations...

    def _check_environment(self):
        """Verify all required dependencies and resources"""
        self._check_gpu_system_readiness() # Renamed and consolidated GPU check
        self._check_memory_requirements() # Assuming this method exists or will be added
        self._verify_python_dependencies() # Assuming this method exists or will be added
        
    def _check_gpu_system_readiness(self):
        """Checks for PyTorch CUDA and RAPIDS libraries and sets unified GPU flags."""
        self.gpu_torch_cuda_available = False
        if _torch_available:
            try:
                if torch.cuda.is_available():
                    self.gpu_torch_cuda_available = True
                    self.logger.info(f"PyTorch CUDA found: True (Device: {torch.cuda.get_device_name(0)})")
                else:
                    self.logger.info("PyTorch found, but torch.cuda.is_available() is False.")
            except Exception as e:
                self.logger.warning(f"Error while checking PyTorch CUDA availability: {e}")
        else:
            self.logger.info("PyTorch not found. Cannot check for Torch CUDA availability.")

        self.gpu_rapids_available = _cupy_cudf_cuml_available
        if self.gpu_rapids_available:
            self.logger.info("RAPIDS libraries (CuPy/cuDF/cuML) found: True")
        else:
            self.logger.info("RAPIDS libraries (CuPy/cuDF/cuML) found: False")

        if self.parameters.use_gpu:
            if self.gpu_torch_cuda_available or self.gpu_rapids_available:
                self.actual_use_gpu = True
                self.logger.info("GPU acceleration is requested and a capable GPU system (Torch CUDA or RAPIDS) is available.")
            else:
                self.actual_use_gpu = False
                self.logger.warning("GPU acceleration requested, but no capable GPU system (Torch CUDA or RAPIDS) was found. Processing will use CPU.")
        else:
            self.actual_use_gpu = False
            self.logger.info("GPU acceleration not requested by parameters. Processing will use CPU.")

    def _check_memory_requirements(self): # Placeholder
        self.logger.debug("Placeholder for _check_memory_requirements")
        try:
            virtual_mem = psutil.virtual_memory()
            available_gb = virtual_mem.available / (1024 ** 3)
            total_gb = virtual_mem.total / (1024 ** 3)
            self.logger.info(f"System Memory: Available {available_gb:.2f} GB / Total {total_gb:.2f} GB")
            if available_gb < 2: # Arbitrary low threshold, e.g. 2GB
                 self.logger.warning("Low available system memory detected. Analysis might be slow or fail.")
        except Exception as e:
            self.logger.warning(f"Could not check system memory: {e}")

    def _verify_python_dependencies(self): # Placeholder
        self.logger.debug("Placeholder for _verify_python_dependencies")
        try:
            import scanpy
            import anndata
            import numpy
            import scipy
            import pandas
            # import rpy2 # Already has try-except block, covered by _rpy2_available
            # import torch # Already has try-except block, covered by _torch_available
            # import cupy # Already has try-except block, covered by _cupy_cudf_cuml_available
            # import scrublet # Covered by _scrublet_available
            # import harmonypy # Covered by _harmonypy_available
            # import gseapy # Covered by _gseapy_available
            # import sqlite_utils # Covered by _sqlite_utils_available
            import matplotlib
            import seaborn
            import sklearn

            self.logger.info(f"Key Python library versions:")
            self.logger.info(f"  scanpy: {scanpy.__version__}")
            self.logger.info(f"  anndata: {anndata.__version__}")
            self.logger.info(f"  numpy: {numpy.__version__}")
            self.logger.info(f"  scipy: {scipy.__version__}")
            self.logger.info(f"  pandas: {pandas.__version__}")
            self.logger.info(f"  matplotlib: {matplotlib.__version__ if _matplotlib_available else 'Not available'}")
            self.logger.info(f"  seaborn: {seaborn.__version__ if _matplotlib_available else 'Not available'}")
            self.logger.info(f"  sklearn: {sklearn.__version__ if _sklearn_silhouette_available else 'Not available (or just metrics parts)'}") # silhouette is the specific check

            # Log availability status of optional packages more explicitly here if desired
            self.logger.info("Status of optional libraries (True = found, False = not found):")
            self.logger.info(f"  PyTorch (for GPU checks/ops): {_torch_available}")
            self.logger.info(f"  RAPIDS (CuPy/cuDF/cuML for GPU ops): {_cupy_cudf_cuml_available}")
            self.logger.info(f"  Scrublet (doublet detection): {_scrublet_available}")
            self.logger.info(f"  rpy2 (for R integration): {_rpy2_available}")
            self.logger.info(f"  Harmonypy (batch correction): {_harmonypy_available}")
            self.logger.info(f"  GSEApy (pathway analysis): {_gseapy_available}")
            self.logger.info(f"  GOATOOLS (pathway analysis): {_goatools_available}")
            self.logger.info(f"  sqlite-utils (database saving): {_sqlite_utils_available}")

        except ImportError as ie:
            self.logger.warning(f"Could not import a core dependency during version check: {ie}. Environment might be incomplete.")
        except Exception as e:
            self.logger.warning(f"Could not verify all Python dependency versions: {e}")
        # This could verify versions of key packages if needed.
        pass

    def _setup_resource_monitor(self):
        """Continuous resource monitoring"""
        self.resource_monitor = ResourceMonitor(
            interval=5,
            track_cpu=True,
            track_memory=True,
            track_gpu=self.actual_use_gpu # Changed from self.gpu_enabled
        )
        self.resource_monitor.start()

class BenchmarkContext:
    """Context manager for benchmarking specific stages of the analysis."""

    def __init__(self, processor: ScRNAProcessor, name: str):
        self.processor = processor
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.processor._track_memory(f"Before {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.processor.benchmarks[self.name + '_time'] = elapsed_time
        self.processor.execution_times[self.name] = elapsed_time
        self.processor.logger.info(f"{self.name} completed in {elapsed_time:.2f} seconds.")
        self.processor._track_memory(f"After {self.name}")

class ResourceMonitor(threading.Thread):
    """Precision resource monitoring thread"""
    def __init__(self, interval=5, **kwargs):
        super().__init__(daemon=True)
        self.interval = interval
        self.tracking = kwargs
        self.data = []
        self.running = True
        
    def run(self):
        while self.running:
            snapshot = {
                'timestamp': time.time(),
                'cpu': psutil.cpu_percent(interval=0.1) if self.tracking.get('track_cpu') else None,
                'memory': psutil.virtual_memory()._asdict() if self.tracking.get('track_memory') else None,
                'gpu': self._get_gpu_stats() if self.tracking.get('track_gpu') else None
            }
            self.data.append(snapshot)
            time.sleep(self.interval)
            
    def _get_gpu_stats(self):
        """Precise GPU monitoring"""
        if _torch_available and self.tracking.get('track_gpu'):
            try:
                if torch.cuda.is_available():
                    return {
                        'utilization': torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'N/A',
                        'memory_used_bytes': torch.cuda.memory_allocated(0),
                        'memory_total_bytes': torch.cuda.get_device_properties(0).total_memory
                    }
                else:
                    return None
            except Exception as e:
                logger.debug(f"Could not get PyTorch GPU stats: {e}")
                return None
        return None
