"""
Custom exceptions for PrecipGen library.

This module provides comprehensive error handling with specific error classes
for different failure modes, detailed error messages, and guidance for users.
"""

from typing import List, Optional, Dict, Any


class PrecipGenError(Exception):
    """
    Base exception for all PrecipGen errors.
    
    Provides structured error information with guidance for resolution.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 guidance: Optional[str] = None):
        """
        Initialize PrecipGen error.
        
        Args:
            message: Primary error message
            details: Additional error details and context
            guidance: Suggested resolution steps
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.guidance = guidance
    
    def __str__(self) -> str:
        """Return formatted error message with details and guidance."""
        result = self.message
        
        if self.details:
            result += f"\nDetails: {self.details}"
        
        if self.guidance:
            result += f"\nGuidance: {self.guidance}"
        
        return result


# Data Loading Errors
class DataLoadError(PrecipGenError):
    """Raised when data cannot be loaded from source."""
    pass


class FileNotFoundError(DataLoadError):
    """Raised when specified data file cannot be found."""
    
    def __init__(self, filepath: str, suggested_paths: Optional[List[str]] = None):
        details = {'filepath': filepath}
        if suggested_paths:
            details['suggested_paths'] = suggested_paths
        
        guidance = "Check that the file path is correct and the file exists."
        if suggested_paths:
            guidance += f" Similar files found: {', '.join(suggested_paths[:3])}"
        
        super().__init__(f"Data file not found: {filepath}", details, guidance)


class ParseError(DataLoadError):
    """Raised when data file cannot be parsed correctly."""
    
    def __init__(self, filepath: str, line_number: Optional[int] = None, 
                 expected_format: Optional[str] = None, parse_details: Optional[str] = None):
        details = {'filepath': filepath}
        if line_number is not None:
            details['line_number'] = line_number
        if expected_format:
            details['expected_format'] = expected_format
        if parse_details:
            details['parse_details'] = parse_details
        
        message = f"Failed to parse data file: {filepath}"
        if line_number is not None:
            message += f" at line {line_number}"
        
        guidance = "Verify that the file format matches the expected structure."
        if expected_format:
            guidance += f" Expected format: {expected_format}"
        
        super().__init__(message, details, guidance)


class UnitConversionError(DataLoadError):
    """Raised when unit conversion encounters unexpected values."""
    
    def __init__(self, value: float, from_unit: str, to_unit: str, 
                 expected_range: Optional[tuple] = None):
        details = {
            'value': value,
            'from_unit': from_unit,
            'to_unit': to_unit
        }
        if expected_range:
            details['expected_range'] = expected_range
        
        message = f"Unit conversion failed: {value} {from_unit} to {to_unit}"
        
        guidance = f"Check that the input value {value} is in the correct units ({from_unit})."
        if expected_range:
            guidance += f" Expected range: {expected_range[0]} to {expected_range[1]} {from_unit}"
        
        super().__init__(message, details, guidance)


# Data Validation Errors
class ValidationError(PrecipGenError):
    """Raised when data validation fails."""
    pass


class DataQualityError(ValidationError):
    """Raised when data quality is insufficient for analysis."""
    
    def __init__(self, quality_issues: List[str], completeness_pct: Optional[float] = None,
                 threshold_pct: Optional[float] = None):
        details = {'quality_issues': quality_issues}
        if completeness_pct is not None:
            details['completeness_percentage'] = completeness_pct
        if threshold_pct is not None:
            details['threshold_percentage'] = threshold_pct
        
        message = f"Data quality insufficient: {', '.join(quality_issues)}"
        
        guidance = "Consider adjusting quality thresholds, using a different time period, "
        guidance += "or preprocessing the data to fill gaps."
        if completeness_pct and threshold_pct:
            guidance += f" Current completeness: {completeness_pct:.1f}%, required: {threshold_pct:.1f}%"
        
        super().__init__(message, details, guidance)


class PhysicalBoundsError(ValidationError):
    """Raised when data values exceed physical bounds."""
    
    def __init__(self, violations: int, min_bound: float, max_bound: float,
                 extreme_values: Optional[List[float]] = None):
        details = {
            'violations': violations,
            'min_bound': min_bound,
            'max_bound': max_bound
        }
        if extreme_values:
            details['extreme_values'] = extreme_values[:5]  # Show first 5
        
        message = f"{violations} values exceed physical bounds [{min_bound}, {max_bound}]"
        
        guidance = "Review the extreme values for data entry errors or consider "
        guidance += "adjusting physical bounds if these are legitimate extreme events."
        
        super().__init__(message, details, guidance)


# Configuration Errors
class ConfigurationError(PrecipGenError):
    """Raised when configuration is invalid."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, validation_errors: List[str], config_section: Optional[str] = None):
        details = {'validation_errors': validation_errors}
        if config_section:
            details['config_section'] = config_section
        
        message = f"Configuration validation failed: {len(validation_errors)} error(s)"
        if config_section:
            message += f" in section '{config_section}'"
        
        guidance = "Review and correct the configuration errors listed in details. "
        guidance += "Refer to the documentation for valid configuration options."
        
        super().__init__(message, details, guidance)


class CompatibilityError(ConfigurationError):
    """Raised when configuration options are incompatible."""
    
    def __init__(self, incompatible_options: List[str], suggested_fix: Optional[str] = None):
        details = {'incompatible_options': incompatible_options}
        if suggested_fix:
            details['suggested_fix'] = suggested_fix
        
        message = f"Incompatible configuration options: {', '.join(incompatible_options)}"
        
        guidance = "These configuration options cannot be used together. "
        if suggested_fix:
            guidance += f"Suggested fix: {suggested_fix}"
        else:
            guidance += "Choose one option or adjust the configuration."
        
        super().__init__(message, details, guidance)


# Parameter Estimation Errors
class ParameterEstimationError(PrecipGenError):
    """Raised when parameter estimation fails."""
    pass


class InsufficientDataError(ParameterEstimationError):
    """Raised when insufficient data is available for parameter estimation."""
    
    def __init__(self, data_points: int, required_points: int, 
                 time_period: Optional[str] = None, parameter_type: Optional[str] = None):
        details = {
            'data_points': data_points,
            'required_points': required_points
        }
        if time_period:
            details['time_period'] = time_period
        if parameter_type:
            details['parameter_type'] = parameter_type
        
        message = f"Insufficient data for parameter estimation: {data_points} points, need {required_points}"
        if time_period:
            message += f" for {time_period}"
        
        guidance = f"Provide at least {required_points} data points for reliable parameter estimation. "
        guidance += "Consider using a longer time period or different data source."
        
        super().__init__(message, details, guidance)


class ConvergenceError(ParameterEstimationError):
    """Raised when parameter fitting algorithms fail to converge."""
    
    def __init__(self, algorithm: str, parameter_type: str, iterations: Optional[int] = None,
                 fallback_available: bool = False):
        details = {
            'algorithm': algorithm,
            'parameter_type': parameter_type,
            'fallback_available': fallback_available
        }
        if iterations is not None:
            details['iterations'] = iterations
        
        message = f"{algorithm} failed to converge for {parameter_type} estimation"
        if iterations is not None:
            message += f" after {iterations} iterations"
        
        guidance = "The fitting algorithm could not find stable parameters. "
        if fallback_available:
            guidance += "Using fallback method or default parameters."
        else:
            guidance += "Try different initial values or a different estimation method."
        
        super().__init__(message, details, guidance)


class TrendAnalysisError(ParameterEstimationError):
    """Raised when trend analysis encounters problems."""
    
    def __init__(self, analysis_type: str, issue: str, affected_parameters: Optional[List[str]] = None):
        details = {
            'analysis_type': analysis_type,
            'issue': issue
        }
        if affected_parameters:
            details['affected_parameters'] = affected_parameters
        
        message = f"Trend analysis failed: {issue} in {analysis_type}"
        
        guidance = "Check that the time series has sufficient variation and length for trend analysis. "
        guidance += "Consider using a different regression method or window size."
        
        super().__init__(message, details, guidance)


# Simulation Errors
class SimulationError(PrecipGenError):
    """Raised when simulation encounters problems."""
    pass


class StateError(SimulationError):
    """Raised when simulation state becomes invalid."""
    
    def __init__(self, state_issue: str, current_state: Optional[Dict[str, Any]] = None,
                 recovery_possible: bool = True):
        details = {'state_issue': state_issue}
        if current_state:
            details['current_state'] = current_state
        details['recovery_possible'] = recovery_possible
        
        message = f"Invalid simulation state: {state_issue}"
        
        guidance = "The simulation state has become invalid. "
        if recovery_possible:
            guidance += "Try resetting the simulation or checking input parameters."
        else:
            guidance += "Restart the simulation with valid initial conditions."
        
        super().__init__(message, details, guidance)


class BoundsError(SimulationError):
    """Raised when parameter drift exceeds physical bounds."""
    
    def __init__(self, parameter: str, value: float, bounds: tuple, 
                 time_step: Optional[int] = None, drift_rate: Optional[float] = None):
        details = {
            'parameter': parameter,
            'value': value,
            'bounds': bounds
        }
        if time_step is not None:
            details['time_step'] = time_step
        if drift_rate is not None:
            details['drift_rate'] = drift_rate
        
        message = f"Parameter {parameter} = {value} exceeds bounds {bounds}"
        if time_step is not None:
            message += f" at time step {time_step}"
        
        guidance = f"Parameter drift has caused {parameter} to exceed physical bounds. "
        guidance += "Consider reducing trend slopes or implementing bounds constraints."
        
        super().__init__(message, details, guidance)


class SynchronizationError(SimulationError):
    """Raised when simulation time becomes out of sync with external systems."""
    
    def __init__(self, internal_time: str, external_time: str, 
                 time_difference: Optional[float] = None):
        details = {
            'internal_time': internal_time,
            'external_time': external_time
        }
        if time_difference is not None:
            details['time_difference_days'] = time_difference
        
        message = f"Time synchronization error: internal={internal_time}, external={external_time}"
        
        guidance = "The simulation's internal clock is out of sync with the external system. "
        guidance += "Reset the simulation or adjust the time synchronization method."
        
        super().__init__(message, details, guidance)


# Engine-specific Errors
class EngineError(PrecipGenError):
    """Base class for engine-specific errors."""
    pass


class BootstrapEngineError(EngineError):
    """Raised when Bootstrap Engine encounters problems."""
    
    def __init__(self, operation: str, issue: str, sampling_mode: Optional[str] = None):
        details = {
            'operation': operation,
            'issue': issue
        }
        if sampling_mode:
            details['sampling_mode'] = sampling_mode
        
        message = f"Bootstrap Engine error in {operation}: {issue}"
        
        guidance = "Check the historical data and sampling configuration. "
        guidance += "Ensure sufficient historical data is available for the selected sampling mode."
        
        super().__init__(message, details, guidance)


class AnalyticalEngineError(EngineError):
    """Raised when Analytical Engine encounters problems."""
    
    def __init__(self, analysis_stage: str, issue: str, 
                 affected_months: Optional[List[int]] = None):
        details = {
            'analysis_stage': analysis_stage,
            'issue': issue
        }
        if affected_months:
            details['affected_months'] = affected_months
        
        message = f"Analytical Engine error in {analysis_stage}: {issue}"
        
        guidance = "Review the input data quality and analysis parameters. "
        guidance += "Consider adjusting the analysis window or data preprocessing."
        
        super().__init__(message, details, guidance)


class SimulationEngineError(EngineError):
    """Raised when Simulation Engine encounters problems."""
    
    def __init__(self, simulation_step: int, issue: str, 
                 current_parameters: Optional[Dict[str, float]] = None):
        details = {
            'simulation_step': simulation_step,
            'issue': issue
        }
        if current_parameters:
            details['current_parameters'] = current_parameters
        
        message = f"Simulation Engine error at step {simulation_step}: {issue}"
        
        guidance = "Check the parameter values and simulation state. "
        guidance += "Verify that all parameters are within valid ranges."
        
        super().__init__(message, details, guidance)


# Utility functions for error handling
def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Create standardized error context information.
    
    Args:
        operation: Name of the operation being performed
        **kwargs: Additional context information
        
    Returns:
        Dictionary with error context
    """
    import datetime
    
    context = {
        'operation': operation,
        'timestamp': datetime.datetime.now().isoformat(),
        'library_version': '1.0.0'  # Should be imported from package
    }
    context.update(kwargs)
    return context


def handle_graceful_degradation(error: Exception, fallback_value: Any, 
                               operation: str, logger=None) -> Any:
    """
    Handle graceful degradation when errors occur.
    
    Args:
        error: The exception that occurred
        fallback_value: Value to return instead
        operation: Description of the operation
        logger: Optional logger for recording the degradation
        
    Returns:
        The fallback value
    """
    if logger:
        logger.warning(f"Graceful degradation in {operation}: {str(error)}")
        logger.warning(f"Using fallback value: {fallback_value}")
    
    return fallback_value


def validate_and_suggest_fixes(validation_errors: List[str], 
                              config_data: Dict[str, Any]) -> List[str]:
    """
    Analyze validation errors and suggest specific fixes.
    
    Args:
        validation_errors: List of validation error messages
        config_data: Configuration data that failed validation
        
    Returns:
        List of suggested fixes
    """
    suggestions = []
    
    for error in validation_errors:
        if "file not found" in error.lower():
            suggestions.append("Check file paths and ensure files exist")
        elif "threshold" in error.lower():
            suggestions.append("Adjust quality thresholds in configuration")
        elif "station" in error.lower():
            suggestions.append("Verify GHCN station IDs are correct (11 characters)")
        elif "directory" in error.lower():
            suggestions.append("Ensure directory paths exist and are accessible")
        elif "completeness" in error.lower():
            suggestions.append("Consider using data with higher completeness or adjust thresholds")
        else:
            suggestions.append("Review configuration documentation for valid options")
    
    return list(set(suggestions))  # Remove duplicates