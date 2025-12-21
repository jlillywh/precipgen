"""
GHCN data parser for PrecipGen library.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
from .base import DataSource


class GHCNParser(DataSource):
    """
    Parser for GHCN .dly format files.
    
    Handles fixed-width GHCN daily format with element codes,
    quality flags, and automatic unit conversion.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize GHCN parser.
        
        Args:
            file_path: Path to GHCN .dly file
        """
        self.file_path = Path(file_path)
        self._validate_file_path()
    
    def _validate_file_path(self) -> None:
        """Validate that the file path exists and is readable."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"GHCN file not found: {self.file_path}")
        
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
        
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"GHCN file is empty: {self.file_path}")
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load and parse GHCN .dly file.
        
        Returns:
            DataFrame with parsed GHCN data
        """
        return self.parse_dly_file(str(self.file_path))
    
    def validate_source(self) -> bool:
        """
        Validate that the GHCN file is accessible and valid.
        
        Returns:
            True if file is valid, False otherwise
        """
        try:
            self._validate_file_path()
            # Try to read first few lines to validate format
            with open(self.file_path, 'r') as f:
                first_line = f.readline().strip()
                if len(first_line) < 21:  # Minimum length for GHCN record
                    return False
            return True
        except Exception:
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the GHCN file.
        
        Returns:
            Dictionary containing file metadata
        """
        return {
            'file_path': str(self.file_path),
            'file_size': self.file_path.stat().st_size,
            'data_type': 'ghcn_dly',
            'station_id': self._extract_station_id()
        }
    
    def _extract_station_id(self) -> str:
        """Extract station ID from filename or first line."""
        # Try filename first (standard GHCN naming)
        filename = self.file_path.stem
        if len(filename) == 11 and filename.isalnum():
            return filename
        
        # Fall back to reading from file
        try:
            with open(self.file_path, 'r') as f:
                first_line = f.readline().strip()
                if len(first_line) >= 11:
                    return first_line[:11]
        except Exception:
            pass
        
        return "UNKNOWN"
    
    def parse_dly_file(self, filepath: str) -> pd.DataFrame:
        """
        Parse GHCN .dly file into structured DataFrame.
        
        Args:
            filepath: Path to .dly file
            
        Returns:
            DataFrame with columns: date, element, value, quality_flag
        """
        records = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) < 21:  # Skip malformed lines
                    continue
                
                # Parse fixed-width format
                station_id = line[0:11]
                year = int(line[11:15])
                month = int(line[15:17])
                element = line[17:21].strip()
                
                # Parse daily values (31 days max)
                for day in range(1, 32):
                    start_pos = 21 + (day - 1) * 8
                    if start_pos + 8 > len(line):
                        break
                    
                    value_str = line[start_pos:start_pos + 5].strip()
                    mflag = line[start_pos + 5:start_pos + 6]  # Measurement flag
                    qflag = line[start_pos + 6:start_pos + 7]  # Quality flag
                    
                    if value_str and value_str != '-9999':
                        try:
                            # Create date, handling invalid dates gracefully
                            try:
                                date = pd.Timestamp(year=year, month=month, day=day)
                            except ValueError:
                                # Skip invalid dates (e.g., Feb 30)
                                continue
                            
                            value = int(value_str)
                            records.append({
                                'station_id': station_id,
                                'date': date,
                                'element': element,
                                'value': value,
                                'quality_flag': qflag.strip()
                            })
                        except ValueError:
                            # Skip malformed values
                            continue
        
        if not records:
            raise ValueError(f"No valid data found in GHCN file: {filepath}")
        
        return pd.DataFrame(records)
    
    def extract_precipitation(self, data: pd.DataFrame) -> pd.Series:
        """
        Extract precipitation data from parsed GHCN DataFrame.
        
        Args:
            data: Parsed GHCN DataFrame
            
        Returns:
            Series with precipitation values indexed by date
        """
        # Filter for precipitation element
        precip_data = data[data['element'] == 'PRCP'].copy()
        
        if precip_data.empty:
            raise ValueError("No precipitation data (PRCP) found in dataset")
        
        # Convert units and create time series
        precip_data['value_mm'] = self.convert_units(precip_data['value'])
        
        # Group by date and take mean if multiple values per day
        precip_series = precip_data.groupby('date')['value_mm'].mean()
        
        return precip_series.sort_index()
    
    def convert_units(self, precip_tenths_mm: pd.Series) -> pd.Series:
        """
        Convert GHCN precipitation from tenths of mm to mm.
        
        Args:
            precip_tenths_mm: Precipitation in tenths of millimeters
            
        Returns:
            Precipitation in millimeters
        """
        # GHCN precipitation is stored in tenths of millimeters
        return precip_tenths_mm / 10.0
    
    def parse_quality_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and categorize GHCN quality flags.
        
        Args:
            data: DataFrame with quality_flag column
            
        Returns:
            DataFrame with additional quality flag analysis
        """
        data = data.copy()
        
        # Define quality flag meanings
        flag_meanings = {
            '': 'Good',
            ' ': 'Good', 
            'D': 'Duplicate',
            'G': 'Gap filled',
            'I': 'Internal consistency check failed',
            'L': 'Lagged range check failed',
            'M': 'Megaconsistency check failed',
            'N': 'Naught check failed',
            'O': 'Climatological outlier check failed',
            'R': 'Lagged range check failed',
            'S': 'Spatial consistency check failed',
            'T': 'Temporal consistency check failed',
            'W': 'Temperature too warm for snow',
            'X': 'Failed bounds check',
            'Z': 'Flagged as a result of an official Datzilla investigation'
        }
        
        data['quality_meaning'] = data['quality_flag'].map(flag_meanings).fillna('Unknown')
        data['is_suspect'] = data['quality_flag'].isin(['X', 'W', 'I', 'O'])
        
        return data