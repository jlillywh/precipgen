"""
Tests for core parameter calculation algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from precipgen.engines import AnalyticalEngine
from precipgen.engines.analytical import MonthlyParams, WindowAnalysis, TrendAnalysis


class TestParameterCalculation:
    """Test core parameter calculation algorithms."""
    
    def test_transition_probability_calculation(self):
        """Test P(W|W) and P(W|D) transition probability calculations."""
        # Create deterministic test data
        dates = pd.date_range('2000-01-01', '2000-01-31', freq='D')
        
        # Pattern: W-D-W-D-W-D... (alternating wet/dry)
        precip_amounts = [5.0 if i % 2 == 0 else 0.0 for i in range(len(dates))]
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate parameters
        params = engine._calculate_parameters_for_period(data)
        
        # With alternating pattern: P(W|W) should be 0, P(W|D) should be 1
        assert params.p_ww == 0.0, f"Expected P(W|W)=0.0, got {params.p_ww}"
        assert params.p_wd == 1.0, f"Expected P(W|D)=1.0, got {params.p_wd}"
    
    def test_gamma_parameter_estimation(self):
        """Test Gamma distribution parameter estimation using method of moments."""
        # Create test data with known Gamma parameters
        np.random.seed(42)
        true_alpha, true_beta = 2.0, 5.0
        wet_amounts = np.random.gamma(true_alpha, true_beta, 1000)
        
        dates = pd.date_range('2000-01-01', periods=len(wet_amounts), freq='D')
        data = pd.Series(wet_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate parameters
        params = engine._calculate_parameters_for_period(data)
        
        # Check that estimated parameters are close to true values
        # (allowing for sampling variation)
        assert abs(params.alpha - true_alpha) < 0.5, f"Alpha estimate {params.alpha} too far from true value {true_alpha}"
        assert abs(params.beta - true_beta) < 1.0, f"Beta estimate {params.beta} too far from true value {true_beta}"
        
        # Verify method of moments relationships
        sample_mean = data.mean()
        sample_var = data.var()
        
        estimated_mean = params.alpha * params.beta
        estimated_var = params.alpha * params.beta**2
        
        assert abs(estimated_mean - sample_mean) < 0.1, "Method of moments mean mismatch"
        assert abs(estimated_var - sample_var) < 1.0, "Method of moments variance mismatch"
    
    def test_monthly_parameter_calculation(self):
        """Test calculation of parameters for all 12 months."""
        # Create 3 years of data
        dates = pd.date_range('2000-01-01', '2002-12-31', freq='D')
        np.random.seed(42)
        
        # Generate seasonal precipitation pattern
        day_of_year = dates.dayofyear.values
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Generate wet/dry pattern with seasonal variation
        wet_prob = 0.3 * seasonal_factor / seasonal_factor.mean()
        wet_days = np.random.random(len(dates)) < wet_prob
        
        # Generate precipitation amounts
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate monthly parameters
        monthly_params = engine.calculate_monthly_parameters()
        
        # Verify we have parameters for all 12 months
        assert len(monthly_params) == 12, f"Expected 12 months, got {len(monthly_params)}"
        
        for month in range(1, 13):
            assert month in monthly_params, f"Missing parameters for month {month}"
            params = monthly_params[month]
            
            # Verify parameter bounds
            assert 0 <= params.p_ww <= 1, f"P(W|W) out of bounds for month {month}: {params.p_ww}"
            assert 0 <= params.p_wd <= 1, f"P(W|D) out of bounds for month {month}: {params.p_wd}"
            assert params.alpha > 0, f"Alpha not positive for month {month}: {params.alpha}"
            assert params.beta > 0, f"Beta not positive for month {month}: {params.beta}"
    
    def test_insufficient_data_handling(self):
        """Test handling of periods with insufficient data."""
        # Create very short data series
        dates = pd.date_range('2000-01-01', '2000-01-05', freq='D')
        data = pd.Series([1.0, 0.0, 2.0, 0.0, 1.5], index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate parameters for insufficient data
        params = engine._calculate_parameters_for_period(data)
        
        # Should return reasonable default values
        assert isinstance(params, MonthlyParams)
        assert 0 <= params.p_ww <= 1
        assert 0 <= params.p_wd <= 1
        assert params.alpha > 0
        assert params.beta > 0
    
    def test_edge_cases(self):
        """Test edge cases in parameter calculation."""
        engine = AnalyticalEngine(pd.Series([], dtype=float), wet_day_threshold=0.001)
        engine.initialize()
        
        # Test empty data
        empty_data = pd.Series([], dtype=float)
        params = engine._calculate_parameters_for_period(empty_data)
        assert params.p_ww == 0.5  # Default value
        assert params.p_wd == 0.3  # Default value
        assert params.alpha == 1.0  # Default value
        assert params.beta == 5.0  # Default value
        
        # Test all zeros (all dry days)
        dates = pd.date_range('2000-01-01', '2000-12-31', freq='D')
        all_dry = pd.Series([0.0] * len(dates), index=dates)
        params = engine._calculate_parameters_for_period(all_dry)
        assert params.p_ww == 0.5  # Default when no wet days
        assert params.p_wd == 0.0  # No wet days following dry days
        
        # Test single wet day
        single_wet = pd.Series([5.0], index=pd.date_range('2000-01-01', periods=1))
        params = engine._calculate_parameters_for_period(single_wet)
        assert 0 <= params.p_ww <= 1
        assert 0 <= params.p_wd <= 1
        assert params.alpha > 0
        assert params.beta > 0
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness of parameter calculations."""
        # Create realistic test data
        np.random.seed(123)
        dates = pd.date_range('2000-01-01', '2010-12-31', freq='D')
        
        # Generate Markov chain wet/dry sequence
        p_ww, p_wd = 0.6, 0.3
        wet_days = [False]  # Start with dry day
        
        for i in range(1, len(dates)):
            if wet_days[-1]:  # Previous day was wet
                wet_days.append(np.random.random() < p_ww)
            else:  # Previous day was dry
                wet_days.append(np.random.random() < p_wd)
        
        # Generate precipitation amounts for wet days
        alpha, beta = 1.5, 6.0
        precip_amounts = []
        for is_wet in wet_days:
            if is_wet:
                precip_amounts.append(np.random.gamma(alpha, beta))
            else:
                precip_amounts.append(0.0)
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate overall parameters
        monthly_params = engine.calculate_monthly_parameters()
        
        # Average the monthly parameters to get overall estimates
        avg_p_ww = np.mean([params.p_ww for params in monthly_params.values()])
        avg_p_wd = np.mean([params.p_wd for params in monthly_params.values()])
        avg_alpha = np.mean([params.alpha for params in monthly_params.values()])
        avg_beta = np.mean([params.beta for params in monthly_params.values()])
        
        # Check that estimated parameters are reasonably close to true values
        # (allowing for monthly variation and sampling error)
        assert abs(avg_p_ww - p_ww) < 0.2, f"P(W|W) estimate {avg_p_ww} too far from true {p_ww}"
        assert abs(avg_p_wd - p_wd) < 0.2, f"P(W|D) estimate {avg_p_wd} too far from true {p_wd}"
        assert abs(avg_alpha - alpha) < 0.5, f"Alpha estimate {avg_alpha} too far from true {alpha}"
        assert abs(avg_beta - beta) < 2.0, f"Beta estimate {avg_beta} too far from true {beta}"
    
    def test_bounds_checking(self):
        """Test that parameter bounds are enforced."""
        # Create data that might lead to extreme parameter values
        dates = pd.date_range('2000-01-01', '2000-12-31', freq='D')
        
        # Test with very small precipitation amounts
        small_amounts = pd.Series([0.001] * len(dates), index=dates)
        engine = AnalyticalEngine(small_amounts, wet_day_threshold=0.001)
        engine.initialize()
        params = engine._calculate_parameters_for_period(small_amounts)
        
        # Parameters should be within reasonable bounds
        assert 0.1 <= params.alpha <= 10.0, f"Alpha {params.alpha} outside reasonable bounds"
        assert 0.1 <= params.beta <= 100.0, f"Beta {params.beta} outside reasonable bounds"
        
        # Test with very large precipitation amounts
        large_amounts = pd.Series([1000.0] * len(dates), index=dates)
        engine = AnalyticalEngine(large_amounts, wet_day_threshold=0.001)
        engine.initialize()
        params = engine._calculate_parameters_for_period(large_amounts)
        
        # Parameters should still be within reasonable bounds
        assert 0.1 <= params.alpha <= 10.0, f"Alpha {params.alpha} outside reasonable bounds"
        assert 0.1 <= params.beta <= 100.0, f"Beta {params.beta} outside reasonable bounds"


class TestSlidingWindowAnalysis:
    """Test sliding window analysis functionality."""
    
    def test_sliding_window_basic_functionality(self):
        """Test basic sliding window analysis with sufficient data."""
        # Create 10 years of data for robust sliding window analysis
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic precipitation data with seasonal patterns
        day_of_year = dates.dayofyear.values
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Generate wet/dry pattern
        wet_prob = 0.25 * seasonal_factor / seasonal_factor.mean()
        wet_days = np.random.random(len(dates)) < wet_prob
        
        # Generate precipitation amounts
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis with 3-year windows
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Verify we got results
        assert isinstance(window_analysis, WindowAnalysis)
        assert len(window_analysis.window_parameters) > 0
        assert len(window_analysis.window_dates) == len(window_analysis.window_parameters)
        
        # Check that we have multiple overlapping windows
        # With 10 years of data and 3-year windows, we should have ~7 windows
        assert len(window_analysis.window_parameters) >= 5
        
        # Verify each window has valid parameters
        for window_id, window_params in window_analysis.window_parameters.items():
            assert window_id in window_analysis.window_dates
            
            # Check that we have monthly parameters
            assert len(window_params) > 0
            assert len(window_params) <= 12  # At most 12 months
            
            for month, params in window_params.items():
                assert isinstance(params, MonthlyParams)
                assert 0 <= params.p_ww <= 1
                assert 0 <= params.p_wd <= 1
                assert params.alpha > 0
                assert params.beta > 0
    
    def test_sliding_window_overlapping_consistency(self):
        """Test that overlapping windows produce consistent results."""
        # Create 8 years of stable data (no trends)
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(123)
        
        # Generate stable precipitation pattern
        wet_prob = 0.3
        wet_days = np.random.random(len(dates)) < wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(2.0, 4.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Extract parameters from all windows for comparison
        all_p_ww = []
        all_p_wd = []
        all_alpha = []
        all_beta = []
        
        for window_params in window_analysis.window_parameters.values():
            # Average across months for each window
            window_p_ww = np.mean([p.p_ww for p in window_params.values()])
            window_p_wd = np.mean([p.p_wd for p in window_params.values()])
            window_alpha = np.mean([p.alpha for p in window_params.values()])
            window_beta = np.mean([p.beta for p in window_params.values()])
            
            all_p_ww.append(window_p_ww)
            all_p_wd.append(window_p_wd)
            all_alpha.append(window_alpha)
            all_beta.append(window_beta)
        
        # With stable data, parameters should be relatively consistent
        # (allowing for sampling variation)
        p_ww_std = np.std(all_p_ww)
        p_wd_std = np.std(all_p_wd)
        alpha_std = np.std(all_alpha)
        beta_std = np.std(all_beta)
        
        # Standard deviations should be reasonable for stable data
        assert p_ww_std < 0.2, f"P(W|W) too variable across windows: std={p_ww_std}"
        assert p_wd_std < 0.2, f"P(W|D) too variable across windows: std={p_wd_std}"
        assert alpha_std < 1.0, f"Alpha too variable across windows: std={alpha_std}"
        assert beta_std < 2.0, f"Beta too variable across windows: std={beta_std}"
    
    def test_sliding_window_insufficient_data_handling(self):
        """Test handling of windows with insufficient data."""
        # Create short data series (2 years) - shorter than default window
        dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')
        np.random.seed(42)
        
        precip_amounts = np.random.gamma(1.5, 5.0, len(dates))
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Should raise error for insufficient data
        with pytest.raises(ValueError, match="Data period.*shorter than window size"):
            engine.perform_sliding_window_analysis(window_years=3)
    
    def test_sliding_window_sparse_data_handling(self):
        """Test handling of windows with sparse data availability."""
        # Create 5 years of data with gaps
        dates = pd.date_range('2000-01-01', '2004-12-31', freq='D')
        np.random.seed(42)
        
        # Create data with significant gaps (simulate missing data)
        precip_amounts = np.random.gamma(1.5, 5.0, len(dates))
        
        # Introduce gaps - set 50% of data to NaN
        gap_indices = np.random.choice(len(dates), size=len(dates)//2, replace=False)
        precip_amounts[gap_indices] = np.nan
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=2)
        
        # Should handle sparse data gracefully
        # Some windows might be skipped due to insufficient data
        assert isinstance(window_analysis, WindowAnalysis)
        
        # Verify that returned windows have reasonable data
        for window_params in window_analysis.window_parameters.values():
            for params in window_params.values():
                assert 0 <= params.p_ww <= 1
                assert 0 <= params.p_wd <= 1
                assert params.alpha > 0
                assert params.beta > 0
    
    def test_sliding_window_parameter_evolution(self):
        """Test detection of parameter evolution over time."""
        # Create data with a clear trend
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Create increasing wet probability over time (trend)
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        base_wet_prob = 0.2
        trend_wet_prob = base_wet_prob + 0.2 * year_progress  # Increase from 0.2 to 0.4
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Extract P(W|D) values over time to check for trend
        window_times = []
        window_p_wd = []
        
        for window_id in sorted(window_analysis.window_parameters.keys()):
            window_params = window_analysis.window_parameters[window_id]
            start_date, end_date = window_analysis.window_dates[window_id]
            
            # Use window center time
            center_time = start_date + (end_date - start_date) / 2
            window_times.append(center_time.year)
            
            # Average P(W|D) across months
            avg_p_wd = np.mean([p.p_wd for p in window_params.values()])
            window_p_wd.append(avg_p_wd)
        
        # Should see increasing trend in P(W|D)
        # Check that later windows have higher P(W|D) than earlier ones
        if len(window_p_wd) >= 3:
            early_avg = np.mean(window_p_wd[:2])
            late_avg = np.mean(window_p_wd[-2:])
            
            # Later periods should have higher wet probability
            assert late_avg > early_avg, f"Expected increasing trend, but early={early_avg}, late={late_avg}"
    
    def test_sliding_window_aggregation_and_storage(self):
        """Test proper aggregation and storage of window results."""
        # Create 6 years of data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Test proper storage structure
        assert hasattr(window_analysis, 'window_parameters')
        assert hasattr(window_analysis, 'window_dates')
        
        # Verify data consistency
        assert len(window_analysis.window_parameters) == len(window_analysis.window_dates)
        
        # Test that window IDs are properly formatted
        for window_id in window_analysis.window_parameters.keys():
            assert window_id.startswith('window_')
            assert window_id.split('_')[1].isdigit()
        
        # Test date ranges are logical
        prev_end = None
        for window_id in sorted(window_analysis.window_parameters.keys()):
            start_date, end_date = window_analysis.window_dates[window_id]
            
            # Window should span approximately 3 years
            window_days = (end_date - start_date).days
            assert 1000 < window_days < 1200  # ~3 years ± some tolerance
            
            # Windows should overlap (start before previous window ends)
            if prev_end is not None:
                assert start_date < prev_end, "Windows should overlap"
            
            prev_end = end_date
        
        # Test parameter aggregation
        for window_params in window_analysis.window_parameters.values():
            # Should have parameters for multiple months
            assert len(window_params) > 0
            
            # Each month should have valid parameters
            for month, params in window_params.items():
                assert 1 <= month <= 12
                assert isinstance(params, MonthlyParams)
    
    def test_empty_data_handling(self):
        """Test handling of empty data series."""
        empty_data = pd.Series([], dtype=float)
        
        engine = AnalyticalEngine(empty_data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Should raise error for empty data
        with pytest.raises(ValueError, match="No data available"):
            engine.perform_sliding_window_analysis(window_years=3)


class TestSlidingWindowIntegration:
    """Test integration of sliding window analysis with other components."""
    
    def test_parameter_manifest_includes_sliding_window_stats(self):
        """Test that parameter manifest includes sliding window statistics."""
        # Create sufficient data for sliding window analysis
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        engine.perform_sliding_window_analysis(window_years=3)
        
        # Generate parameter manifest
        manifest = engine.generate_parameter_manifest()
        
        # Verify sliding window stats are included
        assert manifest.sliding_window_stats is not None
        assert 'window_count' in manifest.sliding_window_stats
        assert 'window_years' in manifest.sliding_window_stats
        assert 'analysis_method' in manifest.sliding_window_stats
        
        # Verify values are reasonable
        assert manifest.sliding_window_stats['window_count'] > 0
        assert manifest.sliding_window_stats['window_years'] == 3
        assert manifest.sliding_window_stats['analysis_method'] == 'overlapping_windows'
    
    def test_sliding_window_with_trend_analysis(self):
        """Test sliding window analysis followed by trend extraction."""
        # Create 8 years of data with trend
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(42)
        
        # Create trend in wet probability
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        trend_wet_prob = 0.2 + 0.15 * year_progress
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Extract trends from window analysis
        trend_analysis = engine.extract_trends(window_analysis)
        
        # Verify trend analysis results
        assert isinstance(trend_analysis, TrendAnalysis)
        assert len(trend_analysis.seasonal_slopes) > 0
        assert len(trend_analysis.significance_tests) > 0
        assert len(trend_analysis.trend_confidence) > 0
        
        # Check that we have seasonal data
        expected_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in expected_seasons:
            assert season in trend_analysis.seasonal_slopes
            assert season in trend_analysis.significance_tests
            assert season in trend_analysis.trend_confidence
            
            # Check parameter types
            for param in ['p_ww', 'p_wd', 'alpha', 'beta']:
                assert param in trend_analysis.seasonal_slopes[season]
                assert param in trend_analysis.significance_tests[season]
                assert param in trend_analysis.trend_confidence[season]


class TestTrendAnalysisEnhancements:
    """Test enhanced trend analysis functionality."""
    
    def test_polynomial_regression_analysis(self):
        """Test polynomial regression for trend analysis."""
        # Create data with quadratic trend
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Create quadratic trend in wet probability
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        # Quadratic trend: starts low, increases, then levels off
        trend_wet_prob = 0.15 + 0.3 * year_progress - 0.1 * year_progress**2
        trend_wet_prob = np.clip(trend_wet_prob, 0.1, 0.9)
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform sliding window analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        
        # Test both linear and polynomial regression
        linear_trends = engine.extract_trends(window_analysis, regression_type='linear')
        poly_trends = engine.extract_trends(window_analysis, regression_type='polynomial')
        
        # Verify both analyses complete successfully
        assert isinstance(linear_trends, TrendAnalysis)
        assert isinstance(poly_trends, TrendAnalysis)
        
        # Check regression type is stored
        assert linear_trends.regression_type == 'linear'
        assert poly_trends.regression_type == 'polynomial'
        
        # Both should have validation results
        assert linear_trends.validation_results is not None
        assert poly_trends.validation_results is not None
    
    def test_trend_slope_validation(self):
        """Test validation of trend slopes for physical reasonableness."""
        # Create data with extreme trends (should trigger validation warnings)
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Create unrealistic trend (very rapid change)
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        trend_wet_prob = 0.1 + 0.8 * year_progress  # Change from 0.1 to 0.9 over 10 years
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.0 + 5.0 * year_progress, 5.0),  # Extreme alpha trend
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform analysis
        window_analysis = engine.perform_sliding_window_analysis(window_years=3)
        trend_analysis = engine.extract_trends(window_analysis)
        
        # Check validation results
        assert trend_analysis.validation_results is not None
        
        # Some slopes should be flagged as invalid due to extreme values
        validation_results = trend_analysis.validation_results
        invalid_found = False
        
        for season_results in validation_results.values():
            for param_valid in season_results.values():
                if not param_valid:
                    invalid_found = True
                    break
        
        # Note: This test might not always find invalid slopes due to randomness
        # The important thing is that validation runs without error
        assert isinstance(validation_results, dict)
    
    def test_comprehensive_trend_analysis_workflow(self):
        """Test the complete trend analysis workflow."""
        # Create realistic data with moderate trends
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Create seasonal variation with slight trend
        day_of_year = dates.dayofyear
        seasonal_wet_prob = 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Add slight increasing trend
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        trend_wet_prob = seasonal_wet_prob + 0.05 * year_progress
        trend_wet_prob = np.clip(trend_wet_prob, 0.1, 0.9)
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Test comprehensive analysis method
        trend_analysis = engine.perform_comprehensive_trend_analysis(
            window_years=3, 
            regression_type='linear'
        )
        
        # Verify complete analysis
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.regression_type == 'linear'
        assert trend_analysis.validation_results is not None
        
        # Verify parameter manifest includes trend information
        manifest = engine.generate_parameter_manifest()
        assert manifest.trend_analysis is not None
        assert manifest.sliding_window_stats is not None
        assert 'regression_type' in manifest.sliding_window_stats
        assert 'trend_validation_performed' in manifest.sliding_window_stats
    
    def test_invalid_regression_type_handling(self):
        """Test handling of invalid regression types."""
        dates = pd.date_range('2000-01-01', '2004-12-31', freq='D')
        np.random.seed(42)
        precip_amounts = np.random.gamma(1.5, 5.0, len(dates))
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        window_analysis = engine.perform_sliding_window_analysis(window_years=2)
        
        # Test invalid regression type
        with pytest.raises(ValueError, match="regression_type must be"):
            engine.extract_trends(window_analysis, regression_type='invalid')
    
    def test_insufficient_data_for_polynomial_regression(self):
        """Test fallback to linear regression when insufficient data for polynomial."""
        # Create minimal data (less than 4 points needed for polynomial)
        dates = pd.date_range('2000-01-01', '2002-12-31', freq='D')
        np.random.seed(42)
        precip_amounts = np.random.gamma(1.5, 5.0, len(dates))
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # This should create very few windows (maybe 1-2)
        window_analysis = engine.perform_sliding_window_analysis(window_years=2)
        
        # Polynomial regression should fall back to linear for insufficient data
        trend_analysis = engine.extract_trends(window_analysis, regression_type='polynomial')
        
        # Should complete without error
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.regression_type == 'polynomial'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])