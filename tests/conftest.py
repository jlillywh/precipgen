"""
Pytest configuration and fixtures for PrecipGen tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import strategies as st


@pytest.fixture
def sample_precipitation_data():
    """Generate sample precipitation time series for testing."""
    dates = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    
    # Generate realistic precipitation data
    np.random.seed(42)
    
    # Create seasonal pattern
    day_of_year = dates.dayofyear.values
    seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Generate wet/dry pattern (30% wet days on average)
    wet_prob = 0.3 * seasonal_factor / seasonal_factor.mean()
    wet_days = np.random.random(len(dates)) < wet_prob
    
    # Generate precipitation amounts for wet days
    precip_amounts = np.where(
        wet_days,
        np.random.gamma(1.5, 5.0, len(dates)),  # Gamma distribution for wet days
        0.0
    )
    
    return pd.Series(precip_amounts, index=dates)


@pytest.fixture
def sample_ghcn_data():
    """Generate sample GHCN-format data for testing."""
    station_id = "USC00123456"
    year = 2020
    month = 1
    element = "PRCP"
    
    # Create a sample GHCN line
    line = f"{station_id}{year:4d}{month:02d}{element}"
    
    # Add 31 daily values (in tenths of mm)
    for day in range(1, 32):
        if day <= 31:  # Valid days
            value = np.random.randint(0, 500)  # 0-50mm in tenths
            quality_flag = " "  # Good quality
            line += f"{value:5d} {quality_flag} "
        else:
            line += "-9999   "  # Missing value
    
    return line.strip()


# Hypothesis strategies for property-based testing
@st.composite
def precipitation_time_series(draw, min_length=100, max_length=1000):
    """Generate realistic precipitation time series."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    start_date = draw(st.dates(
        min_value=datetime(1950, 1, 1).date(),
        max_value=datetime(2020, 1, 1).date()
    ))
    
    dates = pd.date_range(start_date, periods=length, freq='D')
    
    # Generate wet/dry pattern
    wet_prob = draw(st.floats(min_value=0.1, max_value=0.8))
    wet_days = draw(st.lists(
        st.booleans().map(lambda x: x if np.random.random() < wet_prob else False),
        min_size=length,
        max_size=length
    ))
    
    # Generate precipitation amounts
    precip_amounts = []
    for is_wet in wet_days:
        if is_wet:
            amount = draw(st.floats(min_value=0.1, max_value=100.0))
        else:
            amount = 0.0
        precip_amounts.append(amount)
    
    return pd.Series(precip_amounts, index=dates)


@st.composite
def ghcn_station_ids(draw):
    """Generate valid GHCN station identifiers."""
    # GHCN station IDs are 11 characters: country code + network + station
    country_codes = ['US', 'CA', 'MX', 'UK', 'FR', 'DE']
    country = draw(st.sampled_from(country_codes))
    
    # Network code (1 char) + station number (8 chars)
    network = draw(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=1, max_size=1))
    station_num = draw(st.text(alphabet='0123456789', min_size=8, max_size=8))
    
    return country + network + station_num


@st.composite
def parameter_sets(draw):
    """Generate valid WGEN parameter combinations."""
    p_ww = draw(st.floats(min_value=0.0, max_value=1.0))
    p_wd = draw(st.floats(min_value=0.0, max_value=1.0))
    alpha = draw(st.floats(min_value=0.1, max_value=10.0))
    beta = draw(st.floats(min_value=0.1, max_value=100.0))
    
    return {
        'p_ww': p_ww,
        'p_wd': p_wd,
        'alpha': alpha,
        'beta': beta
    }