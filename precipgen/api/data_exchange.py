"""
Standardized data exchange formats for PrecipGen library.

Provides consistent data structures using standard Python types
for seamless integration with external simulation environments.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class SimulationState:
    """
    Standardized simulation state representation.
    
    Uses only standard Python data types for external compatibility.
    """
    current_date: str  # ISO format datetime string
    elapsed_days: int
    is_wet: bool
    random_seed_state: Optional[Dict[str, Any]] = None
    engine_type: str = "unknown"
    additional_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationState':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ParameterSet:
    """
    Standardized parameter set representation.
    
    Uses standard Python data types for external compatibility.
    """
    monthly_parameters: Dict[str, Dict[str, float]]  # month -> parameter -> value
    metadata: Dict[str, Any]
    trend_slopes: Optional[Dict[str, Dict[str, float]]] = None  # season -> parameter -> slope
    validation_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """Create from dictionary."""
        return cls(**data)


class DataExchangeFormat:
    """
    Standardized data exchange format utilities.
    
    Provides methods to convert between internal library formats
    and standard Python data types for external integration.
    """
    
    @staticmethod
    def to_standard_dict(obj: Any) -> Dict[str, Any]:
        """
        Convert any object to standard dictionary format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Dictionary with standard Python data types
        """
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return DataExchangeFormat._convert_dict_values(obj.__dict__)
        elif isinstance(obj, dict):
            return DataExchangeFormat._convert_dict_values(obj)
        else:
            return {'value': DataExchangeFormat._convert_value(obj)}
    
    @staticmethod
    def to_numpy_array(data: Union[List, np.ndarray]) -> np.ndarray:
        """
        Convert data to numpy array format.
        
        Args:
            data: Data to convert
            
        Returns:
            NumPy array
        """
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    @staticmethod
    def from_numpy_array(array: np.ndarray) -> List:
        """
        Convert numpy array to standard Python list.
        
        Args:
            array: NumPy array to convert
            
        Returns:
            Python list
        """
        return array.tolist()
    
    @staticmethod
    def standardize_time_series(data: Any, 
                               dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Standardize time series data for external exchange.
        
        Args:
            data: Time series data (pandas Series, numpy array, or list)
            dates: Optional list of ISO format date strings
            
        Returns:
            Dictionary with standardized time series format
        """
        # Convert data to standard list
        if hasattr(data, 'values'):  # pandas Series
            values = data.values.tolist()
            if dates is None and hasattr(data, 'index'):
                dates = [d.isoformat() if hasattr(d, 'isoformat') else str(d) 
                        for d in data.index]
        elif isinstance(data, np.ndarray):
            values = data.tolist()
        elif isinstance(data, list):
            values = data
        else:
            values = [data]
        
        result = {
            'values': values,
            'length': len(values),
            'data_type': 'time_series'
        }
        
        if dates is not None:
            result['dates'] = dates
            result['has_dates'] = True
        else:
            result['has_dates'] = False
        
        return result
    
    @staticmethod
    def _convert_dict_values(d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary values to standard types."""
        result = {}
        for key, value in d.items():
            # Convert key to string for checking
            key_str = str(key)
            
            # Skip problematic attributes that can cause recursion
            if key_str in ['logger', '_logger', '__dict__', '__class__', '__module__']:
                continue
            # Skip private attributes and methods (only for string keys)
            if isinstance(key, str) and key.startswith('_') and not key.startswith('__'):
                continue
            result[key_str] = DataExchangeFormat._convert_value(value)
        return result
    
    @staticmethod
    def _convert_value(value: Any) -> Any:
        """Convert individual value to standard type."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif hasattr(value, 'to_dict'):
            return value.to_dict()
        elif hasattr(value, '__dict__'):
            # Avoid recursion with logger objects and other problematic types
            if hasattr(value, '__class__') and 'logger' in str(value.__class__).lower():
                return f"<Logger: {value.__class__.__name__}>"
            return DataExchangeFormat._convert_dict_values(value.__dict__)
        elif isinstance(value, dict):
            return DataExchangeFormat._convert_dict_values(value)
        elif isinstance(value, (list, tuple)):
            return [DataExchangeFormat._convert_value(item) for item in value]
        else:
            return str(value)


class ValidationReport:
    """
    Standardized validation report format.
    
    Provides consistent error and warning reporting across all components.
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_valid: bool = True
    
    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add info message."""
        self.info.append(message)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = DataExchangeFormat._convert_value(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external use."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'metadata': self.metadata,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info)
        }