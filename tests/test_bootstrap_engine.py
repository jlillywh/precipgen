"""
Comprehensive tests for BootstrapEngine functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from precipgen.engines import BootstrapEngine


class TestBootstrapEngine:
    """Test BootstrapEngine functionality against requirements."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical precipitation data spanning multiple years."""
        # Create 5 years of daily data (2000-2004)
        dates = pd.date_range('2000-01-01', '2004-12-31', freq='D')
        
        # Create realistic precipitation data with seasonal patterns
        np.random.seed(42)  # For reproducible tests
        precip_values = []
        
        for date in dates:
            # Higher precipitation in winter months
            if date.month in [12, 1, 2]:
                base_prob = 0.4
                base_amount = 8.0
            elif date.month in [6, 7, 8]:  # Summer - lower precipitation
                base_prob = 0.1
                base_amount = 3.0
            else:  # Spring/Fall
                base_prob = 0.25
                base_amount = 5.0
            
            # Generate wet/dry day
            if np.random.random() < base_prob:
                # Wet day - sample from gamma distribution
                amount = np.random.gamma(1.5, base_amount)
            else:
                # Dry day
                amount = 0.0
            
            precip_values.append(amount)
        
        return pd.Series(precip_values, index=dates)
    
    @pytest.fixture
    def leap_year_data(self):
        """Create data spanning leap and non-leap years."""
        # Include 2000 (leap) and 2001 (non-leap)
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        np.random.seed(123)
        values = np.random.gamma(1.5, 5.0, len(dates))
        return pd.Series(values, index=dates)
    
    def test_bootstrap_engine_initialization(self, sample_historical_data):
        """Test requirement 3.1: Bootstrap_Engine initializes and loads complete historical record."""
        engine = BootstrapEngine(sample_historical_data, mode='random')
        
        # Verify initialization
        assert engine.historical_data is not None
        assert len(engine.historical_data) == len(sample_historical_data)
        assert engine.mode == 'random'
        assert engine.available_years == [2000, 2001, 2002, 2003, 2004]
        
        # Test initialization with start date
        start_date = datetime(2010, 1, 1)
        engine.initialize(start_date, random_seed=42)
        
        assert engine.current_date == start_date
        assert engine.random_state is not None
        assert engine.current_year_data is not None
    
    def test_random_sampling_mode(self, sample_historical_data):
        """Test requirement 3.2: Random sampling mode randomly selects historical years."""
        engine = BootstrapEngine(sample_historical_data, mode='random')
        engine.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Collect years sampled over multiple year transitions
        sampled_years = []
        current_date = datetime(2010, 1, 1)
        
        # Run for 3 years to see year transitions
        for _ in range(365 * 3):
            if current_date.month == 1 and current_date.day == 1:
                sampled_years.append(engine.get_current_year())
            
            precip = engine.step()
            current_date += timedelta(days=1)
            assert isinstance(precip, float)
            assert precip >= 0.0
        
        # Should have sampled multiple different years
        unique_years = set(sampled_years)
        assert len(unique_years) > 1  # Should have variety in random sampling
        
        # All sampled years should be from available historical years
        for year in unique_years:
            assert year in [2000, 2001, 2002, 2003, 2004]
    
    def test_sequential_sampling_mode(self, sample_historical_data):
        """Test requirement 3.3: Sequential sampling mode cycles through years in order with wraparound."""
        engine = BootstrapEngine(sample_historical_data, mode='sequential')
        engine.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Track year progression
        year_sequence = []
        current_date = datetime(2010, 1, 1)
        
        # Record initial year
        year_sequence.append(engine.get_current_year())
        
        # Run for 7 years to see wraparound (5 historical years + 2 more)
        for day in range(365 * 7):
            precip = engine.step()
            current_date += timedelta(days=1)
            assert isinstance(precip, float)
            assert precip >= 0.0
            
            # Record year on Jan 1 after year transition
            if current_date.month == 1 and current_date.day == 1:
                year_sequence.append(engine.get_current_year())
        
        # Should follow sequential pattern: 2000, 2001, 2002, 2003, 2004, 2000, 2001
        expected_sequence = [2000, 2001, 2002, 2003, 2004, 2000, 2001]
        assert year_sequence == expected_sequence
    
    def test_leap_year_handling(self, leap_year_data):
        """Test requirement 3.4: Leap year handling for year transitions."""
        engine = BootstrapEngine(leap_year_data, mode='sequential')
        
        # Test non-leap year simulation with leap historical year
        # Simulate 2001 (non-leap) using 2000 (leap) historical data
        engine.initialize(datetime(2001, 2, 27), random_seed=42)
        
        # Step through Feb 28
        precip_feb27 = engine.step()  # Feb 27 -> Feb 28
        assert engine.current_date == datetime(2001, 2, 28)
        
        # Step through Feb 28 -> Mar 1 (Feb 29 doesn't exist in 2001)
        precip_feb28 = engine.step()  # Feb 28 -> Mar 1
        assert engine.current_date == datetime(2001, 3, 1)
        
        # The precipitation should be valid
        assert isinstance(precip_feb28, float)
        assert precip_feb28 >= 0.0
        
        # Test leap year simulation with leap historical data
        # Reset and test leap year simulation (2000 is leap year)
        engine.reset(datetime(2000, 2, 27))
        
        precip_feb27 = engine.step()  # Feb 27 -> Feb 28
        assert engine.current_date == datetime(2000, 2, 28)
        
        precip_feb28 = engine.step()  # Feb 28 -> Feb 29
        assert engine.current_date == datetime(2000, 2, 29)
        
        precip_feb29 = engine.step()  # Feb 29 -> Mar 1
        assert engine.current_date == datetime(2000, 3, 1)
        
        # All precipitation values should be valid
        assert isinstance(precip_feb28, float)
        assert isinstance(precip_feb29, float)
        assert precip_feb28 >= 0.0
        assert precip_feb29 >= 0.0
    
    def test_long_simulation_wraparound(self, sample_historical_data):
        """Test requirement 3.5: Seamless wraparound for long simulations."""
        engine = BootstrapEngine(sample_historical_data, mode='sequential')
        engine.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Run a very long simulation (10 years) to test wraparound
        total_days = 365 * 10
        precipitation_values = []
        
        for _ in range(total_days):
            precip = engine.step()
            precipitation_values.append(precip)
            assert isinstance(precip, float)
            assert precip >= 0.0
        
        # Should have completed without errors
        assert len(precipitation_values) == total_days
        
        # Verify state is maintained properly
        state = engine.get_state()
        assert 'current_date' in state
        assert 'current_position' in state
        assert 'sequential_year_index' in state
        assert 'current_historical_year' in state
    
    def test_state_management(self, sample_historical_data):
        """Test state management and reset functionality."""
        engine = BootstrapEngine(sample_historical_data, mode='random')
        engine.initialize(datetime(2010, 6, 15), random_seed=42)
        
        # Get initial state
        initial_state = engine.get_state()
        assert initial_state['current_date'] == datetime(2010, 6, 15)
        
        # Step forward several days
        for _ in range(10):
            engine.step()
        
        # State should have changed
        new_state = engine.get_state()
        assert new_state['current_date'] != initial_state['current_date']
        
        # Reset to original date
        engine.reset(datetime(2010, 6, 15))
        reset_state = engine.get_state()
        assert reset_state['current_date'] == datetime(2010, 6, 15)
    
    def test_error_handling(self, sample_historical_data):
        """Test error handling for invalid operations."""
        engine = BootstrapEngine(sample_historical_data, mode='random')
        
        # Should raise error if step() called before initialize()
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            engine.step()
        
        # Should raise error for invalid mode
        with pytest.raises(ValueError, match="Unknown sampling mode"):
            BootstrapEngine(sample_historical_data, mode='invalid_mode')
    
    def test_empty_data_handling(self):
        """Test handling of empty historical data."""
        empty_data = pd.Series([], dtype=float)
        
        with pytest.raises(ValueError, match="No historical data available"):
            BootstrapEngine(empty_data, mode='random')
    
    def test_missing_value_handling(self, sample_historical_data):
        """Test handling of missing values in historical data."""
        # Introduce some NaN values
        data_with_nans = sample_historical_data.copy()
        data_with_nans.iloc[100:110] = np.nan
        
        engine = BootstrapEngine(data_with_nans, mode='random')
        engine.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Should handle NaN values gracefully (convert to 0.0)
        for _ in range(50):
            precip = engine.step()
            assert not np.isnan(precip)
            assert precip >= 0.0
    
    def test_reproducibility(self, sample_historical_data):
        """Test that results are reproducible with same random seed."""
        # Create two engines with same seed
        engine1 = BootstrapEngine(sample_historical_data, mode='random')
        engine2 = BootstrapEngine(sample_historical_data, mode='random')
        
        engine1.initialize(datetime(2010, 1, 1), random_seed=42)
        engine2.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Generate same sequence
        values1 = [engine1.step() for _ in range(100)]
        values2 = [engine2.step() for _ in range(100)]
        
        # Should be identical
        assert values1 == values2
    
    def test_different_modes_produce_different_results(self, sample_historical_data):
        """Test that random and sequential modes produce different results."""
        engine_random = BootstrapEngine(sample_historical_data, mode='random')
        engine_sequential = BootstrapEngine(sample_historical_data, mode='sequential')
        
        engine_random.initialize(datetime(2010, 1, 1), random_seed=42)
        engine_sequential.initialize(datetime(2010, 1, 1), random_seed=42)
        
        # Generate sequences
        random_values = [engine_random.step() for _ in range(365 * 2)]
        sequential_values = [engine_sequential.step() for _ in range(365 * 2)]
        
        # Should be different (very unlikely to be identical by chance)
        assert random_values != sequential_values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])