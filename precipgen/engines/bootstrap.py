"""
Bootstrap engine for historical resampling.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .base import Engine


class BootstrapEngine(Engine):
    """
    Historical resampling engine with random or sequential modes.
    
    Provides bootstrap sampling of historical precipitation data
    with support for different sampling strategies and leap year handling.
    """
    
    def __init__(self, historical_data: pd.Series, mode: str = 'random'):
        """
        Initialize bootstrap engine.
        
        Args:
            historical_data: Time series of historical precipitation data
            mode: Sampling mode ('random' or 'sequential')
        """
        if mode not in ['random', 'sequential']:
            raise ValueError(f"Unknown sampling mode: {mode}. Must be 'random' or 'sequential'")
        
        self.historical_data = historical_data.copy()
        self.mode = mode
        self.random_state = None
        self.current_date = None
        self.current_year_data = None
        self.current_position = 0
        self.available_years = None
        self.sequential_year_index = 0
        self._current_historical_year = None
        
        self._prepare_historical_data()
    
    def _prepare_historical_data(self) -> None:
        """Prepare historical data for sampling."""
        if len(self.historical_data) == 0:
            raise ValueError("No historical data available for bootstrap sampling")
        
        # Ensure data is sorted by date
        self.historical_data = self.historical_data.sort_index()
        
        # Extract available years - handle case where index might not be datetime
        if hasattr(self.historical_data.index, 'year'):
            self.available_years = sorted(self.historical_data.index.year.unique())
        else:
            raise ValueError("Historical data must have a datetime index")
        
        if len(self.available_years) == 0:
            raise ValueError("No historical data available for bootstrap sampling")
    
    def initialize(self, start_date: datetime, random_seed: Optional[int] = None) -> None:
        """
        Initialize bootstrap engine for sampling.
        
        Args:
            start_date: Starting date for simulation
            random_seed: Random seed for reproducible sampling
        """
        self.current_date = start_date
        self.random_state = np.random.RandomState(random_seed)
        self.sequential_year_index = 0
        self._last_sim_year = start_date.year
        
        # Load initial year data
        self._load_year_data()
    
    def reset(self, start_date: Optional[datetime] = None) -> None:
        """
        Reset engine to initial state.
        
        Args:
            start_date: Optional new start date
        """
        if start_date:
            self.current_date = start_date
        
        self.current_year_data = None
        self.current_position = 0
        self.sequential_year_index = 0
        
        if self.current_date:
            self._load_year_data()
    
    def step(self) -> float:
        """
        Generate next daily precipitation value.
        
        Returns:
            Daily precipitation in millimeters
        """
        if self.current_date is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Get precipitation for current date
        precip_value = self._get_current_precipitation()
        
        # Advance to next day
        self._advance_day()
        
        # Check for year transition after advancing
        self._check_year_transition()
        
        return precip_value
    
    def _check_year_transition(self) -> None:
        """Check if we need to advance to the next historical year."""
        if (self.current_date.month == 1 and self.current_date.day == 1):
            current_sim_year = self.current_date.year
            if current_sim_year > self._last_sim_year:
                # New year detected, advance to next historical year
                if self.mode == 'sequential':
                    self.sequential_year_index += 1
                self._load_year_data()
            self._last_sim_year = current_sim_year
    
    def get_current_year(self) -> int:
        """
        Get the historical year currently being sampled.
        
        Returns:
            Year from historical data being used
        """
        if hasattr(self, '_current_historical_year'):
            return self._current_historical_year
        elif self.current_year_data is not None and len(self.current_year_data) > 0:
            return self.current_year_data.index[0].year
        else:
            return None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current engine state.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'mode': self.mode,
            'current_date': self.current_date,
            'current_position': self.current_position,
            'sequential_year_index': self.sequential_year_index,
            'current_historical_year': self.get_current_year(),
            'available_years_count': len(self.available_years)
        }
    
    def _load_year_data(self) -> None:
        """Load data for the current simulation year."""
        if self.mode == 'random':
            historical_year = self.random_state.choice(self.available_years)
        elif self.mode == 'sequential':
            index = self.sequential_year_index % len(self.available_years)
            historical_year = self.available_years[index]
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")
        
        # Extract year data
        year_mask = self.historical_data.index.year == historical_year
        self.current_year_data = self.historical_data[year_mask].copy()
        
        # Reset position for new year
        self.current_position = 0
        
        # Handle leap year transitions
        self._handle_leap_year_transition()
        
        # Store the historical year for get_current_year()
        self._current_historical_year = historical_year
    
    def _handle_leap_year_transition(self) -> None:
        """Handle transitions between leap and non-leap years."""
        current_year = self.current_date.year
        historical_year = self.current_year_data.index[0].year
        
        current_is_leap = self._is_leap_year(current_year)
        historical_is_leap = self._is_leap_year(historical_year)
        
        # If simulation year is leap but historical isn't, duplicate Feb 28 for Feb 29
        if current_is_leap and not historical_is_leap:
            feb_28_mask = (
                (self.current_year_data.index.month == 2) & 
                (self.current_year_data.index.day == 28)
            )
            if feb_28_mask.any():
                feb_28_value = self.current_year_data[feb_28_mask].iloc[0]
                # Create Feb 29 entry using a leap year for the timestamp
                # We'll use 2000 as a reference leap year
                feb_29_date = pd.Timestamp(year=2000, month=2, day=29)
                feb_29_series = pd.Series([feb_28_value], index=[feb_29_date])
                self.current_year_data = pd.concat([self.current_year_data, feb_29_series])
                self.current_year_data = self.current_year_data.sort_index()
        
        # If simulation year is not leap but historical is, remove Feb 29
        elif not current_is_leap and historical_is_leap:
            feb_29_mask = (
                (self.current_year_data.index.month == 2) & 
                (self.current_year_data.index.day == 29)
            )
            self.current_year_data = self.current_year_data[~feb_29_mask]
    
    def _get_current_precipitation(self) -> float:
        """Get precipitation value for current date."""
        # Find the data entry that matches the current simulation date
        matching_dates = self.current_year_data.index[
            (self.current_year_data.index.month == self.current_date.month) &
            (self.current_year_data.index.day == self.current_date.day)
        ]
        
        if len(matching_dates) == 0:
            # No matching date found (e.g., Feb 29 in non-leap year)
            # Return 0.0 for missing dates
            return 0.0
        
        # Get value for the matching date
        precip_value = self.current_year_data.loc[matching_dates[0]]
        
        # Handle missing values
        if pd.isna(precip_value):
            precip_value = 0.0
        
        return float(precip_value)
    
    def _advance_day(self) -> None:
        """Advance to next day in simulation."""
        next_date = self.current_date + timedelta(days=1)
        
        # Handle leap year transitions - if next date would be Feb 29 in a non-leap year, skip to Mar 1
        if (next_date.month == 2 and next_date.day == 29 and 
            not self._is_leap_year(next_date.year)):
            next_date = next_date.replace(month=3, day=1)
        
        self.current_date = next_date
        self.current_position += 1
    
    def _is_leap_year(self, year: int) -> bool:
        """Check if year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)