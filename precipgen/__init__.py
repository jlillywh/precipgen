"""
PrecipGen: A modular stochastic weather generation library.

This library provides tools for generating synthetic daily precipitation data
using established meteorological algorithms, with emphasis on maintaining state
between calls for seamless integration into dynamic system simulations.
"""

__version__ = "0.2.0"
__author__ = "PrecipGen Development Team"

# Core components
from .engines.bootstrap import BootstrapEngine
from .engines.analytical import AnalyticalEngine
from .engines.simulation import SimulationEngine
from .config.precipgen_config import PrecipGenConfig
from .config.quality_config import QualityConfig
from .config.quality_presets import QualityPresets
from .data.ghcn_parser import GHCNParser
from .data.validator import DataValidator
from .data.ghcn_downloader import GHCNDownloader, find_nearby_stations, download_station
from .simulation.monte_carlo import PrecipitationSimulator
from .simulation.analysis import DroughtAnalyzer, ExtremeEventAnalyzer, SeasonalAnalyzer

__all__ = [
    "BootstrapEngine",
    "AnalyticalEngine", 
    "SimulationEngine",
    "PrecipGenConfig",
    "QualityConfig",
    "QualityPresets",
    "GHCNParser",
    "DataValidator",
    "GHCNDownloader",
    "find_nearby_stations",
    "download_station",
    "PrecipitationSimulator",
    "DroughtAnalyzer",
    "ExtremeEventAnalyzer", 
    "SeasonalAnalyzer",
]