"""
Utility functions and classes for PrecipGen library.

This module provides common utilities, exceptions, and helper functions
used throughout the library.
"""

from .exceptions import (
    PrecipGenError,
    DataLoadError,
    ValidationError,
    ConfigurationError,
    ParameterEstimationError,
    FileNotFoundError,
    ParseError,
    UnitConversionError,
    DataQualityError,
    PhysicalBoundsError,
    ConfigValidationError,
    CompatibilityError,
    InsufficientDataError,
    ConvergenceError,
    TrendAnalysisError,
    SimulationError,
    StateError,
    BoundsError,
    SynchronizationError,
    EngineError,
    BootstrapEngineError,
    AnalyticalEngineError,
    SimulationEngineError,
    create_error_context,
    handle_graceful_degradation,
    validate_and_suggest_fixes
)

from .logging_config import (
    get_logger,
    configure_logging,
    PrecipGenLogger,
    log_operation_start,
    log_operation_complete
)

__all__ = [
    # Base exceptions
    "PrecipGenError",
    "DataLoadError", 
    "ValidationError",
    "ConfigurationError",
    "ParameterEstimationError",
    # Data loading errors
    "FileNotFoundError",
    "ParseError",
    "UnitConversionError",
    # Validation errors
    "DataQualityError",
    "PhysicalBoundsError",
    # Configuration errors
    "ConfigValidationError",
    "CompatibilityError",
    # Parameter estimation errors
    "InsufficientDataError",
    "ConvergenceError",
    "TrendAnalysisError",
    # Simulation errors
    "SimulationError",
    "StateError",
    "BoundsError",
    "SynchronizationError",
    # Engine errors
    "EngineError",
    "BootstrapEngineError",
    "AnalyticalEngineError",
    "SimulationEngineError",
    # Error handling utilities
    "create_error_context",
    "handle_graceful_degradation",
    "validate_and_suggest_fixes",
    # Logging
    "get_logger",
    "configure_logging",
    "PrecipGenLogger",
    "log_operation_start",
    "log_operation_complete",
]