"""
Data source configuration for PrecipGen library.
"""

from typing import List, Optional
from pathlib import Path


class DataSourceConfig:
    """
    Configuration for individual data sources.
    
    Handles file paths and GHCN station management.
    """
    
    def __init__(self, 
                 file_path: Optional[str] = None,
                 station_id: Optional[str] = None,
                 data_type: str = 'csv',
                 **kwargs):
        """
        Initialize data source configuration.
        
        Args:
            file_path: Path to data file
            station_id: GHCN station identifier
            data_type: Type of data file ('csv', 'ghcn_dly')
            **kwargs: Additional configuration parameters
        """
        self.file_path = file_path
        self.station_id = station_id
        self.data_type = data_type
        self.additional_params = kwargs
    
    def validate(self) -> List[str]:
        """
        Validate data source configuration.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate file path
        if self.file_path:
            file_path = Path(self.file_path)
            if not file_path.exists():
                errors.append(f"Data file does not exist: {self.file_path}")
            elif not file_path.is_file():
                errors.append(f"Data path is not a file: {self.file_path}")
            elif not file_path.stat().st_size > 0:
                errors.append(f"Data file is empty: {self.file_path}")
        
        # Validate station ID format for GHCN data
        if self.station_id:
            if len(self.station_id) != 11:
                errors.append(f"GHCN station ID must be 11 characters: {self.station_id}")
            if not self.station_id.isalnum():
                errors.append(f"GHCN station ID must be alphanumeric: {self.station_id}")
        
        # Validate data type
        valid_types = ['csv', 'ghcn_dly']
        if self.data_type not in valid_types:
            errors.append(f"Invalid data_type '{self.data_type}'. Must be one of: {valid_types}")
        
        # For GHCN data, either file_path or station_id should be provided
        if self.data_type == 'ghcn_dly':
            if not self.file_path and not self.station_id:
                errors.append("Either file_path or station_id must be provided for GHCN data")
        
        return errors