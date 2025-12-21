"""
Abstract base classes for data sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    Defines the interface that all data sources must implement
    to maintain separation between data management and algorithms.
    """
    
    @abstractmethod
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataLoadError: If data cannot be loaded
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that the data source is accessible and valid.
        
        Returns:
            True if source is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        
        Returns:
            Dictionary containing source metadata
        """
        pass