"""
Configuration management for PrecipGen library.

This module provides configuration classes for managing dataset paths,
parameters, and operational modes.
"""

from .precipgen_config import PrecipGenConfig
from .data_source_config import DataSourceConfig
from .quality_config import QualityConfig

__all__ = [
    "PrecipGenConfig",
    "DataSourceConfig", 
    "QualityConfig",
]