"""
Precipitation simulation module for PrecipGen.

This module provides high-level simulation classes for Monte Carlo
precipitation modeling with comprehensive analysis capabilities.
"""

from .monte_carlo import PrecipitationSimulator
from .analysis import DroughtAnalyzer, ExtremeEventAnalyzer, SeasonalAnalyzer

__all__ = [
    'PrecipitationSimulator',
    'DroughtAnalyzer', 
    'ExtremeEventAnalyzer',
    'SeasonalAnalyzer'
]