"""
Standardized API interfaces for PrecipGen library.

This module provides standardized interfaces and data exchange formats
for seamless integration with external simulation environments.
"""

from .standardized_api import StandardizedAPI
from .data_exchange import DataExchangeFormat, SimulationState, ParameterSet
from .synchronization import ExternalSimulationSync

__all__ = [
    "StandardizedAPI",
    "DataExchangeFormat", 
    "SimulationState",
    "ParameterSet",
    "ExternalSimulationSync",
]