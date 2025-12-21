"""
Simulation engines for PrecipGen library.

This module provides the core engines for precipitation generation:
- Bootstrap engine for historical resampling
- Analytical engine for parameter extraction
- Simulation engine for synthetic generation
"""

from .base import Engine
from .bootstrap import BootstrapEngine
from .analytical import AnalyticalEngine
from .simulation import SimulationEngine

__all__ = [
    "Engine",
    "BootstrapEngine",
    "AnalyticalEngine", 
    "SimulationEngine",
]