"""
PrecipGen: A modular stochastic weather generation library.

This library provides tools for generating synthetic daily precipitation data
using established meteorological algorithms, with emphasis on maintaining state
between calls for seamless integration into dynamic system simulations.
"""

__version__ = "0.1.1"
__author__ = "PrecipGen Development Team"

# Core components
from .engines.bootstrap import BootstrapEngine
from .engines.analytical import AnalyticalEngine
from .engines.simulation import SimulationEngine
from .config.precipgen_config import PrecipGenConfig
from .config.quality_config import QualityConfig
from .data.ghcn_parser import GHCNParser
from .data.validator import DataValidator

__all__ = [
    "BootstrapEngine",
    "AnalyticalEngine", 
    "SimulationEngine",
    "PrecipGenConfig",
    "QualityConfig",
    "GHCNParser",
    "DataValidator",
]