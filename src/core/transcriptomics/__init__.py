"""
Single-cell RNA-sequencing analysis module for OpenCell.

This module provides tools for processing and analyzing single-cell RNA-seq data, 
including preprocessing, dimensionality reduction, clustering, and visualization.
"""

from .sc_rna_processor import ScRNAProcessor, ScRNAParameters

__all__ = ["ScRNAProcessor", "ScRNAParameters"]
