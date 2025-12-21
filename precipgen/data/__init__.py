"""
Data management components for PrecipGen library.

This module provides classes for loading, parsing, and validating
climate data from various sources including GHCN format files.
"""

from .ghcn_parser import GHCNParser
from .validator import DataValidator, ValidationResult, QualityReport
from .base import DataSource
from .ghcn_downloader import GHCNDownloader

__all__ = [
    "GHCNParser",
    "DataValidator",
    "ValidationResult", 
    "QualityReport",
    "DataSource",
    "GHCNDownloader",
]