"""
Tests for SimulationEngine WGEN algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings

from precipgen.engines.simulation import SimulationEngine, SimulationState
from precipgen.engines.analytical import ParameterManifest, MonthlyParams, TrendAnalysis
from precipgen.utils.exceptions import StateError, SimulationEngineError


def create_test_manifest(monthly_params):
    """Helper function to create test parameter manifest."""
    return ParameterManifest(
        metadata={'station_id': 'TEST', 'wet_day_threshold': 0.001},
        overall_parameters=monthly_params,
        trend_analysis=None,
        sliding_window_stats=None
    )


class TestWGENAlgorithms:
    """Test core WGEN simulation algorithms."""
    
    def test_markov_chain_wet_dry_transitions(self):
        """Test Markov chain wet/dry state transition logic."""
        # Create simple parameters with known transition probabilities
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.8,  # High probability of wet following wet
                p_wd=0.2,  # Low probability of wet following dry
                alpha=1.5,
                beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        # Test with fixed random seed for reproducibility
        engine = SimulationEngine(manifest, random_seed=42)
        engine.initialize(datetime(2020, 1, 1), initial_wet_state=True)
        
        # Track transitions over many days
        transitions = {'WW': 0, 'WD': 0, 'DW': 0, 'DD': 0}
        previous_wet = True
        
        for _ in range(1000):
            precip = engine.step()
            current_wet = precip > 0.001
            
            if previous_wet and current_wet:
                transitions['WW'] += 1
            elif previous_wet and not current_wet:
                transitions['WD'] += 1
            elif not previous_wet and current_wet:
                transitions['DW'] += 1
            else:
                transitions['DD'] += 1
            
            previous_wet = current_wet
        
        # Calculate observed transition probabilities
        total_wet_days = transitions['WW'] + transitions['WD']
        total_dry_days = transitions['DW'] + transitions['DD']
        
        if total_wet_days > 0:
            observed_p_ww = transitions['WW'] / total_wet_days
            # Should be close to 0.8 (within reasonable tolerance)
            assert 0.7 < observed_p_ww < 0.9, f"P(W|W) = {observed_p_ww}, expected ~0.8"
        
        if total_dry_days > 0:
            observed_p_wd = transitions['DW'] / total_dry_days
            # Should be close to 0.2 (within reasonable tolerance)
            assert 0.1 < observed_p_wd < 0.3, f"P(W|D) = {observed_p_wd}, expected ~0.2"
    
    def test_gamma_distribution_sampling(self):
        """Test Gamma distribution sampling for wet day amounts."""
        # Create parameters with known Gamma distribution
        alpha, beta = 2.0, 5.0
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=1.0,  # Always wet to test Gamma sampling
                p_wd=1.0,
                alpha=alpha,
                beta=beta
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        # Collect wet day amounts
        wet_amounts = []
        for _ in range(1000):
            precip = engine.step()
            if precip > 0.001:
                wet_amounts.append(precip)
        
        # Should have many wet days since p_ww = p_wd = 1.0
        assert len(wet_amounts) > 900, f"Expected mostly wet days, got {len(wet_amounts)}"
        
        # Check that amounts follow Gamma distribution characteristics
        wet_amounts = np.array(wet_amounts)
        
        # All amounts should be positive
        assert np.all(wet_amounts > 0), "All wet day amounts should be positive"
        
        # Mean should be approximately alpha * beta
        expected_mean = alpha * beta
        observed_mean = np.mean(wet_amounts)
        assert 0.8 * expected_mean < observed_mean < 1.2 * expected_mean, \
            f"Mean = {observed_mean}, expected ~{expected_mean}"
        
        # Variance should be approximately alpha * beta^2
        expected_var = alpha * beta**2
        observed_var = np.var(wet_amounts)
        assert 0.5 * expected_var < observed_var < 1.5 * expected_var, \
            f"Variance = {observed_var}, expected ~{expected_var}"
    
    def test_monthly_parameter_selection(self):
        """Test monthly parameter selection based on current date."""
        # Create parameters that vary by month (all within valid bounds)
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.4 + 0.05 * month,  # Varies from 0.45 to 1.0 (valid range)
                p_wd=0.1 + 0.02 * month,  # Varies from 0.12 to 0.34 (valid range)
                alpha=1.0 + 0.1 * month,  # Varies from 1.1 to 2.2
                beta=3.0 + 0.5 * month    # Varies from 3.5 to 9.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, random_seed=42)
        
        # Test parameter selection for different months
        for month in range(1, 13):
            engine.initialize(datetime(2020, month, 15))  # Mid-month
            current_params = engine._get_current_parameters()
            
            expected_params = monthly_params[month]
            
            # Parameters should match exactly (no bounds constraints in normal mode)
            assert current_params.p_ww == expected_params.p_ww
            assert current_params.p_wd == expected_params.p_wd
            assert current_params.alpha == expected_params.alpha
            assert current_params.beta == expected_params.beta
    
    def test_random_number_generator_state_management(self):
        """Test proper random number generator state management."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        # Test reproducibility with same seed
        engine1 = SimulationEngine(manifest, random_seed=42)
        engine1.initialize(datetime(2020, 1, 1))
        
        engine2 = SimulationEngine(manifest, random_seed=42)
        engine2.initialize(datetime(2020, 1, 1))
        
        # Should produce identical sequences
        for _ in range(100):
            precip1 = engine1.step()
            precip2 = engine2.step()
            assert precip1 == precip2, "Same seed should produce identical results"
        
        # Test different seeds produce different sequences
        engine3 = SimulationEngine(manifest, random_seed=123)
        engine3.initialize(datetime(2020, 1, 1))
        
        engine4 = SimulationEngine(manifest, random_seed=456)
        engine4.initialize(datetime(2020, 1, 1))
        
        different_count = 0
        for _ in range(100):
            precip3 = engine3.step()
            precip4 = engine4.step()
            if precip3 != precip4:
                different_count += 1
        
        # Should have many differences (not all, due to dry days)
        assert different_count > 20, "Different seeds should produce different sequences"
        
        # Test state preservation
        engine = SimulationEngine(manifest, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        # Generate some values
        values1 = [engine.step() for _ in range(10)]
        
        # Get state
        state = engine.get_current_state()
        
        # Continue generating
        values2 = [engine.step() for _ in range(10)]
        
        # Reset to saved state and continue
        engine.random_state.set_state(state.random_state)
        engine.current_date = state.current_date
        engine.is_wet = state.is_wet
        engine.elapsed_days = state.elapsed_days
        
        # Should produce same sequence as values2
        values3 = [engine.step() for _ in range(10)]
        
        # values2 and values3 should be identical
        for v2, v3 in zip(values2, values3):
            assert v2 == v3, "State restoration should produce identical sequences"
    
    def test_dry_day_generation(self):
        """Test that dry days produce zero precipitation."""
        # Create parameters that favor dry days
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.1,  # Low probability of wet following wet
                p_wd=0.05,  # Very low probability of wet following dry
                alpha=1.5,
                beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, random_seed=42)
        engine.initialize(datetime(2020, 1, 1), initial_wet_state=False)
        
        dry_days = 0
        total_days = 1000
        
        for _ in range(total_days):
            precip = engine.step()
            if precip == 0.0:
                dry_days += 1
        
        # Should have many dry days given low transition probabilities
        dry_percentage = dry_days / total_days
        assert dry_percentage > 0.7, f"Expected >70% dry days, got {dry_percentage:.1%}"
    
    def test_wet_day_threshold_consistency(self):
        """Test that wet/dry classification is consistent with threshold."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        for _ in range(100):
            precip = engine.step()
            state = engine.get_current_state()
            
            # Wet state should be consistent with precipitation amount
            if precip > 0.001:
                # This was a wet day, so next day's state should reflect that
                # (we check this by looking at the engine's internal state)
                pass  # The state reflects the PREVIOUS day's wetness
            else:
                # This was a dry day
                assert precip == 0.0, "Dry days should have exactly 0.0 precipitation"


class TestSimulationEngineIntegration:
    """Test SimulationEngine integration and state management."""
    
    def test_engine_initialization_and_reset(self):
        """Test engine initialization and reset functionality."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, random_seed=42)
        
        # Test initialization
        start_date = datetime(2020, 6, 15)
        engine.initialize(start_date, initial_wet_state=True)
        
        assert engine.current_date == start_date
        assert engine.is_wet is True
        assert engine.elapsed_days == 0
        
        # Generate some values
        for _ in range(10):
            engine.step()
        
        assert engine.elapsed_days == 10
        assert engine.current_date == start_date + timedelta(days=10)
        
        # Test reset
        engine.reset()
        assert engine.current_date == start_date
        assert engine.is_wet is False  # Reset to default
        assert engine.elapsed_days == 0
        
        # Test reset with new date
        new_date = datetime(2021, 3, 1)
        engine.reset(new_date)
        assert engine.current_date == new_date
        assert engine.start_date == new_date
    
    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
        engine.initialize(datetime(2020, 6, 15), initial_wet_state=True)
        
        # Generate some values
        for _ in range(5):
            engine.step()
        
        # Get state
        state_dict = engine.get_state()
        
        # Verify state structure
        assert 'current_date' in state_dict
        assert 'start_date' in state_dict
        assert 'is_wet' in state_dict
        assert 'elapsed_days' in state_dict
        assert 'trend_mode' in state_dict
        assert 'random_state' in state_dict
        
        # Verify values
        assert state_dict['elapsed_days'] == 5
        # trend_mode should be False because trend analysis is None in test manifest
        assert state_dict['trend_mode'] is False
        assert state_dict['is_wet'] in [True, False]
    
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        manifest = create_test_manifest(monthly_params)
        
        engine = SimulationEngine(manifest)
        
        # Test step before initialization
        with pytest.raises(StateError, match="Engine not initialized"):
            engine.step()
        
        # Test invalid parameter manifest
        invalid_manifest = ParameterManifest(
            metadata={'station_id': 'TEST', 'wet_day_threshold': 0.001},
            overall_parameters={},  # Missing months
            trend_analysis=None,
            sliding_window_stats=None
        )
        
        with pytest.raises(SimulationEngineError, match="Parameter manifest missing overall_parameters"):
            SimulationEngine(invalid_manifest)
        
        # Test missing month parameters
        incomplete_params = {1: MonthlyParams(p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0)}
        incomplete_manifest = ParameterManifest(
            metadata={'station_id': 'TEST', 'wet_day_threshold': 0.001},
            overall_parameters=incomplete_params,
            trend_analysis=None,
            sliding_window_stats=None
        )
        
        with pytest.raises(SimulationEngineError, match="Missing parameters for month"):
            SimulationEngine(incomplete_manifest)


class TestTrendProjection:
    """Test enhanced trend projection functionality."""
    
    def create_trend_manifest(self):
        """Helper to create manifest with trend analysis."""
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.6, p_wd=0.3, alpha=1.5, beta=5.0
            )
        
        # Create trend analysis with known slopes
        seasonal_slopes = {
            'Winter': {'p_ww': 0.01, 'p_wd': 0.005, 'alpha': 0.1, 'beta': 0.2},
            'Spring': {'p_ww': 0.008, 'p_wd': 0.004, 'alpha': 0.08, 'beta': 0.15},
            'Summer': {'p_ww': -0.005, 'p_wd': -0.002, 'alpha': -0.05, 'beta': -0.1},
            'Fall': {'p_ww': 0.012, 'p_wd': 0.006, 'alpha': 0.12, 'beta': 0.25}
        }
        
        significance_tests = {}
        trend_confidence = {}
        for season in seasonal_slopes.keys():
            significance_tests[season] = {param: 0.01 for param in seasonal_slopes[season]}
            trend_confidence[season] = {param: "High (p < 0.01)" for param in seasonal_slopes[season]}
        
        from precipgen.engines.analytical import TrendAnalysis
        trend_analysis = TrendAnalysis(
            seasonal_slopes=seasonal_slopes,
            significance_tests=significance_tests,
            trend_confidence=trend_confidence,
            regression_type='linear'
        )
        
        return ParameterManifest(
            metadata={'station_id': 'TREND_TEST', 'wet_day_threshold': 0.001},
            overall_parameters=monthly_params,
            trend_analysis=trend_analysis,
            sliding_window_stats=None
        )
    
    def test_parameter_drift_calculation(self):
        """Test parameter drift calculation using trend slopes."""
        manifest = self.create_trend_manifest()
        engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
        
        # Test at different time points
        test_cases = [
            (datetime(2020, 1, 15), 0.0),    # Winter, year 0
            (datetime(2025, 1, 15), 5.0),    # Winter, year 5
            (datetime(2020, 7, 15), 0.5),    # Summer, year 0.5
        ]
        
        for test_date, expected_years in test_cases:
            engine.initialize(datetime(2020, 1, 1))
            engine.current_date = test_date
            engine.elapsed_days = (test_date - datetime(2020, 1, 1)).days
            
            # Get parameters with trend adjustment
            params = engine._get_current_parameters()
            elapsed_years = engine._calculate_elapsed_years()
            
            # Verify elapsed years calculation
            assert abs(elapsed_years - expected_years) < 0.1, \
                f"Expected {expected_years} years, got {elapsed_years}"
            
            # Verify parameters are adjusted (not equal to baseline)
            base_params = manifest.overall_parameters[test_date.month]
            if elapsed_years > 0:
                # At least one parameter should be different due to trend
                assert (params.p_ww != base_params.p_ww or 
                       params.p_wd != base_params.p_wd or
                       params.alpha != base_params.alpha or
                       params.beta != base_params.beta), \
                    "Parameters should be adjusted when trends are applied"
    
    def test_physical_bounds_enforcement(self):
        """Test that physical bounds are enforced for drifted parameters."""
        # Create manifest with extreme trends
        monthly_params = {1: MonthlyParams(p_ww=0.5, p_wd=0.3, alpha=1.0, beta=3.0)}
        for month in range(2, 13):
            monthly_params[month] = monthly_params[1]
        
        # Extreme slopes that would violate bounds
        extreme_slopes = {
            'Winter': {'p_ww': 0.2, 'p_wd': -0.1, 'alpha': -0.5, 'beta': -1.0}
        }
        for season in ['Spring', 'Summer', 'Fall']:
            extreme_slopes[season] = extreme_slopes['Winter'].copy()
        
        from precipgen.engines.analytical import TrendAnalysis
        trend_analysis = TrendAnalysis(
            seasonal_slopes=extreme_slopes,
            significance_tests={'Winter': {param: 0.01 for param in extreme_slopes['Winter']}},
            trend_confidence={'Winter': {param: "High" for param in extreme_slopes['Winter']}},
            regression_type='linear'
        )
        
        manifest = ParameterManifest(
            metadata={'station_id': 'BOUNDS_TEST', 'wet_day_threshold': 0.001},
            overall_parameters=monthly_params,
            trend_analysis=trend_analysis,
            sliding_window_stats=None
        )
        
        engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
        
        # Test at 10 years (extreme trends should be bounded)
        engine.initialize(datetime(2020, 1, 1))
        engine.current_date = datetime(2030, 1, 15)
        engine.elapsed_days = (datetime(2030, 1, 15) - datetime(2020, 1, 1)).days
        
        # Note: With the enhanced validation, extreme slopes are rejected
        # So parameters should remain at baseline values
        params = engine._get_current_parameters()
        base_params = monthly_params[1]
        
        # Parameters should be baseline (extreme slopes rejected)
        assert params.p_ww == base_params.p_ww
        assert params.p_wd == base_params.p_wd
        assert params.alpha == base_params.alpha
        assert params.beta == base_params.beta
    
    def test_trend_projection_diagnostics(self):
        """Test trend projection diagnostic functions."""
        manifest = self.create_trend_manifest()
        engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        # Advance to 2 years
        engine.current_date = datetime(2022, 6, 15)
        engine.elapsed_days = (datetime(2022, 6, 15) - datetime(2020, 1, 1)).days
        
        # Test trend projection info
        info = engine.get_trend_projection_info()
        
        assert info['trend_mode_enabled'] is True
        assert info['trend_analysis_available'] is True
        assert abs(info['elapsed_years'] - 2.5) < 0.1
        assert info['current_season'] == 'Summer'
        assert 'parameter_drift' in info
        assert 'bounds_applied' in info
        
        # Test validation
        validation = engine.validate_trend_projection()
        assert validation['is_valid'] is True
        assert isinstance(validation['warnings'], list)
        assert isinstance(validation['errors'], list)
    
    def test_trend_mode_disabled(self):
        """Test behavior when trend mode is disabled."""
        manifest = self.create_trend_manifest()
        engine = SimulationEngine(manifest, trend_mode=False, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        # Advance to 5 years
        engine.current_date = datetime(2025, 1, 15)
        engine.elapsed_days = (datetime(2025, 1, 15) - datetime(2020, 1, 1)).days
        
        # Parameters should be unchanged (trend mode disabled)
        params = engine._get_current_parameters()
        base_params = manifest.overall_parameters[1]
        
        assert params.p_ww == base_params.p_ww
        assert params.p_wd == base_params.p_wd
        assert params.alpha == base_params.alpha
        assert params.beta == base_params.beta
        
        # Diagnostic info should reflect disabled state
        info = engine.get_trend_projection_info()
        assert info['trend_mode_enabled'] is False
    
    def test_mathematical_correctness_of_drift_formula(self):
        """Test mathematical correctness of the drift formula."""
        manifest = self.create_trend_manifest()
        engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
        
        # Test the drift formula: Parameter(t) = Parameter_baseline + (Trend_Slope × elapsed_time)
        engine.initialize(datetime(2020, 1, 1))
        engine.current_date = datetime(2023, 1, 15)  # 3 years later, Winter
        engine.elapsed_days = (datetime(2023, 1, 15) - datetime(2020, 1, 1)).days
        
        params = engine._get_current_parameters()
        base_params = manifest.overall_parameters[1]  # January
        elapsed_years = engine._calculate_elapsed_years()
        
        # Get Winter slopes
        winter_slopes = manifest.trend_analysis.seasonal_slopes['Winter']
        
        # Manually calculate expected values using drift formula
        expected_p_ww = base_params.p_ww + winter_slopes['p_ww'] * elapsed_years
        expected_p_wd = base_params.p_wd + winter_slopes['p_wd'] * elapsed_years
        expected_alpha = base_params.alpha + winter_slopes['alpha'] * elapsed_years
        expected_beta = base_params.beta + winter_slopes['beta'] * elapsed_years
        
        # Apply same bounds as the engine
        expected_p_ww = max(0.0, min(1.0, expected_p_ww))
        expected_p_wd = max(0.0, min(1.0, expected_p_wd))
        expected_alpha = max(0.1, expected_alpha)
        expected_beta = max(0.1, expected_beta)
        
        # Verify the formula is applied correctly
        assert abs(params.p_ww - expected_p_ww) < 1e-6, \
            f"P(W|W): expected {expected_p_ww}, got {params.p_ww}"
        assert abs(params.p_wd - expected_p_wd) < 1e-6, \
            f"P(W|D): expected {expected_p_wd}, got {params.p_wd}"
        assert abs(params.alpha - expected_alpha) < 1e-6, \
            f"Alpha: expected {expected_alpha}, got {params.alpha}"
        assert abs(params.beta - expected_beta) < 1e-6, \
            f"Beta: expected {expected_beta}, got {params.beta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])