"""
Data quality configuration for PrecipGen library.
"""

from typing import List


class QualityConfig:
    """
    Configuration for data quality thresholds and validation rules.
    """
    
    def __init__(self,
                 max_missing_percentage: float = 10.0,
                 min_years_required: int = 10,
                 max_consecutive_missing_days: int = 30,
                 physical_bounds_min: float = 0.0,
                 physical_bounds_max: float = 1000.0,  # mm per day
                 quality_flags_to_reject: List[str] = None):
        """
        Initialize quality configuration.
        
        Args:
            max_missing_percentage: Maximum percentage of missing data allowed
            min_years_required: Minimum years of data required for analysis
            max_consecutive_missing_days: Maximum consecutive missing days allowed
            physical_bounds_min: Minimum physically plausible precipitation (mm)
            physical_bounds_max: Maximum physically plausible precipitation (mm)
            quality_flags_to_reject: GHCN quality flags that should reject data
        """
        self.max_missing_percentage = max_missing_percentage
        self.min_years_required = min_years_required
        self.max_consecutive_missing_days = max_consecutive_missing_days
        self.physical_bounds_min = physical_bounds_min
        self.physical_bounds_max = physical_bounds_max
        self.quality_flags_to_reject = quality_flags_to_reject if quality_flags_to_reject is not None else ['X', 'W']
    
    def validate(self) -> List[str]:
        """
        Validate quality configuration.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate percentages
        if not 0 <= self.max_missing_percentage <= 100:
            errors.append("max_missing_percentage must be between 0 and 100")
        
        # Validate minimum years
        if self.min_years_required < 1:
            errors.append("min_years_required must be at least 1")
        
        # Validate consecutive missing days
        if self.max_consecutive_missing_days < 1:
            errors.append("max_consecutive_missing_days must be at least 1")
        
        # Validate physical bounds
        if self.physical_bounds_min < 0:
            errors.append("physical_bounds_min cannot be negative")
        
        if self.physical_bounds_max <= self.physical_bounds_min:
            errors.append("physical_bounds_max must be greater than physical_bounds_min")
        
        return errors