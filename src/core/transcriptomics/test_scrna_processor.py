import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml # For checking parameter save/load
import os # For file checks

# Assuming sc_rna_processor.py is in the same directory or PYTHONPATH is set
from .sc_rna_processor import ScRNAProcessor, ScRNAParameters, _sqlite_utils_available # Import the flag

# Setup basic logging to see output from the processor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_adata(n_cells=100, n_genes=500):
    """Creates a dummy AnnData object for testing."""
    logger.info(f"Creating dummy AnnData with {n_cells} cells and {n_genes} genes.")
    counts = np.random.poisson(5, size=(n_cells, n_genes)) # Using Poisson for more realistic counts
    adata = ad.AnnData(X=counts.astype(np.float32)) # Ensure float32 for some scanpy ops
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    
    # Add some dummy batch info for testing batch correction
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], size=n_cells)
    # Add some dummy mitochondrial genes for pct_counts_mt calculation
    mito_genes_count = min(50, n_genes // 10)
    if mito_genes_count > 0:
        mito_genes = np.random.choice(adata.var_names, size=mito_genes_count, replace=False)
        adata.var['mt'] = adata.var_names.isin(mito_genes)
    else:
        adata.var['mt'] = False # Handle case with very few genes
        
    logger.info("Dummy AnnData created.")
    return adata

def test_scrna_processor_pipeline():
    logger.info("Starting ScRNAProcessor test pipeline...")
    
    output_dir = Path("./scrna_test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 0. Test Parameter Save/Load
        logger.info("Testing ScRNAParameters save/load...")
        original_params = ScRNAParameters(n_pcs=42, umap_min_dist=0.25, use_gpu=False)
        params_file = output_dir / "test_params.yaml"
        original_params.save_to_yaml(params_file)
        assert params_file.exists(), "Parameter file was not created."
        
        loaded_params = ScRNAParameters.load_from_yaml(params_file)
        assert loaded_params.n_pcs == 42, "Loaded n_pcs does not match."
        assert loaded_params.umap_min_dist == 0.25, "Loaded umap_min_dist does not match."
        assert loaded_params == original_params, "Loaded parameters do not match original."
        logger.info("ScRNAParameters save/load test successful.")

        # 1. Load Data (PBMC3k or Dummy)
        try:
            logger.info("Attempting to load sc.datasets.pbmc3k_processed()")
            adata = sc.datasets.pbmc3k_processed()
            if not adata.var_names.is_unique:
                adata.var_names_make_unique()
            # pbmc3k_processed is log-normalized. Create a 'counts' layer if not present for robustness.
            if 'counts' not in adata.layers and adata.X is not None:
                 logger.info("Creating 'counts' layer from .X in pbmc3k_processed for test purposes (assuming .X is lognorm counts).")
                 adata.layers['counts'] = adata.X.copy() # Not ideal but for test consistency
            dataset_id = "pbmc3k_test"
        except Exception as e:
            logger.warning(f"Could not load pbmc3k_processed (Error: {e}). Falling back to dummy data.")
            adata = create_dummy_adata(n_cells=150, n_genes=750) # Slightly larger dummy data
            dataset_id = "dummy_data_test"

        # 2. Define/Update Parameters for the main run
        params = ScRNAParameters(
            min_genes=20,
            min_cells=2,
            max_pct_mito=40.0,
            n_pcs=25,
            n_neighbors=8,
            resolution=0.6,
            use_doublet_detection=False, # Keep False for CI/basic tests without all deps
            batch_correction= ('batch' in adata.obs.columns), # Enable if batch key exists
            batch_key='batch' if ('batch' in adata.obs.columns) else None,
            normalization_method="log1p",
            hvg_method="seurat_v3",
            n_hvgs=200,
            save_intermediates=False, # Keep False for faster test run
            cell_cycle_scoring=True # Test this path
        )
        
        # Add 'batch' and 'mt' if using pbmc3k and they are missing, for consistent testing
        if dataset_id == "pbmc3k_test":
            if 'batch' not in adata.obs:
                logger.info("Adding dummy 'batch' to pbmc3k for testing batch correction.")
                adata.obs['batch'] = np.random.choice(['pbmc_b1', 'pbmc_b2'], size=adata.n_obs)
                params.batch_correction = True
                params.batch_key = 'batch'
            if 'mt' not in adata.var:
                logger.info("Adding dummy 'mt' column to pbmc3k var for testing.")
                mito_genes_count = min(50, adata.n_vars // 10)
                if mito_genes_count > 0:
                    mito_genes = np.random.choice(adata.var_names, size=mito_genes_count, replace=False)
                    adata.var['mt'] = adata.var_names.isin(mito_genes)
                else:
                    adata.var['mt'] = False


        # 3. Instantiate the processor
        processor = ScRNAProcessor(parameters=params)

        # 4. Run the analysis
        logger.info(f"Running the analysis pipeline for {dataset_id}...")
        processed_adata = processor.run_analysis(adata.copy()) # Pass a copy to avoid modifying original test data
        logger.info("Analysis pipeline finished.")

        # 5. Assertions on processed_adata
        assert processed_adata is not None, "Processed AnnData object is None."
        assert processed_adata.n_obs == adata.n_obs, "Number of cells changed during processing."
        # HVG selection reduces n_vars
        assert processed_adata.n_vars <= adata.n_vars, "Number of genes increased during processing."
        assert processed_adata.n_vars == params.n_hvgs, f"Number of genes after HVG selection is not {params.n_hvgs}"


        assert 'X_pca' in processed_adata.obsm, "X_pca not found in processed_adata.obsm."
        assert processed_adata.obsm['X_pca'].shape[1] == params.n_pcs, f"PCA components number mismatch. Expected {params.n_pcs}, Got {processed_adata.obsm['X_pca'].shape[1]}"
        
        cluster_key = params.clustering_method
        assert cluster_key in processed_adata.obs, f"Clustering key '{cluster_key}' not found in processed_adata.obs."
        
        assert 'X_umap' in processed_adata.obsm, "X_umap not found in processed_adata.obsm."
        
        if params.cell_cycle_scoring:
            assert 'phase' in processed_adata.obs, "Cell cycle 'phase' not found in processed_adata.obs."

        logger.info("Basic assertions on processed AnnData passed.")

        # 6. Save results
        logger.info(f"Saving results to {output_dir} with dataset_id: {dataset_id}")
        db_file = output_dir / f"{dataset_id}_analysis_metadata.db"
        save_metadata = processor.save_results(dataset_id=dataset_id, output_path=str(output_dir), db_path=str(db_file))
        logger.info(f"Results saved. Metadata: {save_metadata}")

        # 7. Check Output File Creation
        expected_adata_path = output_dir / f"{dataset_id}_processed_adata.h5ad"
        expected_results_json_path = output_dir / f"{dataset_id}_analysis_results.json"
        expected_times_json_path = output_dir / f"{dataset_id}_execution_times.json"

        assert expected_adata_path.exists(), f"Output AnnData file not found: {expected_adata_path}"
        assert expected_results_json_path.exists(), f"Output results JSON file not found: {expected_results_json_path}"
        assert expected_times_json_path.exists(), f"Output execution times JSON file not found: {expected_times_json_path}"
        if _sqlite_utils_available: # Check db file only if library was available
             assert db_file.exists(), f"SQLite DB file not found: {db_file}"
        logger.info("Output file creation check passed.")

        # 8. Print some outputs
        logger.info("--- Analysis Summary ---")
        logger.info(f"Processed AnnData: {processed_adata.n_obs} cells, {processed_adata.n_vars} genes")
        if "n_clusters" in processor.results:
            logger.info(f"Number of clusters found: {processor.results['n_clusters']}")
        if "silhouette_score" in processor.results and processor.results["silhouette_score"] is not None:
            logger.info(f"Silhouette score: {processor.results['silhouette_score']:.4f}")
        
        logger.info("--- Execution Times (seconds) ---")
        for step, duration in processor.execution_times.items():
            logger.info(f"- {step}: {duration:.2f}s")
        
        logger.info("--- Benchmarks (seconds) ---") # Benchmarks might be a bit redundant if execution_times covers it
        for benchmark_name, duration in processor.benchmarks.items():
             if duration is not None: # Ensure duration is not None
                logger.info(f"- {benchmark_name}: {duration:.2f}s")


        logger.info("Test pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Test pipeline failed with error: {e}", exc_info=True)
        raise
    finally:
        # Optional: Clean up created files
        # import shutil
        # if output_dir.exists():
        #     logger.info(f"Cleaning up test output directory: {output_dir}")
        #     shutil.rmtree(output_dir)
        pass


if __name__ == "__main__":
    logger.info("Starting ScRNAProcessor test script.")
    test_scrna_processor_pipeline()
    logger.info("ScRNAProcessor test script finished.") 