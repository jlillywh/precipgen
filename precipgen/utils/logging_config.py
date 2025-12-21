"""
Logging configuration for PrecipGen library.

Provides standardized logging setup with appropriate levels and formatting
for different components of the library.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


class PrecipGenLogger:
    """
    Centralized logging configuration for PrecipGen library.
    
    Provides consistent logging setup across all library components
    with appropriate formatting and level management.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger for a specific component.
        
        Args:
            name: Logger name (typically module name)
            level: Optional logging level override
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.configure_logging()
        
        if name not in cls._loggers:
            logger = logging.getLogger(f"precipgen.{name}")
            
            if level:
                logger.setLevel(getattr(logging, level.upper()))
            
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def configure_logging(cls, level: str = "INFO", 
                         log_file: Optional[str] = None,
                         format_string: Optional[str] = None) -> None:
        """
        Configure logging for the entire library.
        
        Args:
            level: Default logging level
            log_file: Optional file to write logs to
            format_string: Custom format string for log messages
        """
        if cls._configured:
            return
        
        # Default format string
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(funcName)s:%(lineno)d - %(message)s"
            )
        
        # Configure root logger for precipgen
        root_logger = logging.getLogger("precipgen")
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def set_level(cls, level: str, component: Optional[str] = None) -> None:
        """
        Set logging level for entire library or specific component.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            component: Optional specific component name
        """
        if component:
            if component in cls._loggers:
                cls._loggers[component].setLevel(getattr(logging, level.upper()))
        else:
            # Set level for all loggers
            root_logger = logging.getLogger("precipgen")
            root_logger.setLevel(getattr(logging, level.upper()))
            
            for logger in cls._loggers.values():
                logger.setLevel(getattr(logging, level.upper()))
    
    @classmethod
    def log_error_with_context(cls, logger: logging.Logger, error: Exception,
                              operation: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with full context information.
        
        Args:
            logger: Logger instance to use
            error: Exception that occurred
            operation: Description of the operation that failed
            context: Additional context information
        """
        error_msg = f"Error in {operation}: {str(error)}"
        
        if hasattr(error, 'details') and error.details:
            error_msg += f" | Details: {error.details}"
        
        if context:
            error_msg += f" | Context: {context}"
        
        logger.error(error_msg, exc_info=True)
    
    @classmethod
    def log_warning_with_guidance(cls, logger: logging.Logger, warning: str,
                                 guidance: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning with guidance for resolution.
        
        Args:
            logger: Logger instance to use
            warning: Warning message
            guidance: Optional guidance for resolving the issue
            context: Additional context information
        """
        warning_msg = warning
        
        if guidance:
            warning_msg += f" | Guidance: {guidance}"
        
        if context:
            warning_msg += f" | Context: {context}"
        
        logger.warning(warning_msg)
    
    @classmethod
    def log_performance_metrics(cls, logger: logging.Logger, operation: str,
                               duration: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics for operations.
        
        Args:
            logger: Logger instance to use
            operation: Name of the operation
            duration: Duration in seconds
            metrics: Additional performance metrics
        """
        perf_msg = f"Performance - {operation}: {duration:.3f}s"
        
        if metrics:
            metric_strs = [f"{k}={v}" for k, v in metrics.items()]
            perf_msg += f" | Metrics: {', '.join(metric_strs)}"
        
        logger.info(perf_msg)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Configured logger instance
    """
    return PrecipGenLogger.get_logger(name, level)


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Convenience function to configure library logging.
    
    Args:
        level: Default logging level
        log_file: Optional file to write logs to
    """
    PrecipGenLogger.configure_logging(level, log_file)


def log_operation_start(logger: logging.Logger, operation: str, 
                       parameters: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the start of an operation with parameters.
    
    Args:
        logger: Logger instance
        operation: Operation name
        parameters: Operation parameters
    """
    msg = f"Starting {operation}"
    if parameters:
        param_strs = [f"{k}={v}" for k, v in parameters.items()]
        msg += f" with parameters: {', '.join(param_strs)}"
    
    logger.info(msg)


def log_operation_complete(logger: logging.Logger, operation: str,
                          duration: Optional[float] = None,
                          results: Optional[Dict[str, Any]] = None) -> None:
    """
    Log the completion of an operation with results.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Optional operation duration
        results: Operation results summary
    """
    msg = f"Completed {operation}"
    
    if duration is not None:
        msg += f" in {duration:.3f}s"
    
    if results:
        result_strs = [f"{k}={v}" for k, v in results.items()]
        msg += f" | Results: {', '.join(result_strs)}"
    
    logger.info(msg)