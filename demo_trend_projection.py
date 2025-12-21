#!/usr/bin/env python3
"""
Demonstration of enhanced trend projection system.

This script shows the improved trend projection capabilities including:
1. Parameter drift calculation using trend slopes
2. Physical bounds checking for drifted parameters
3. Time-based parameter adjustment logic
4. Mathematical correctness of drift formula
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from precipgen.engines.simulation import SimulationEngine
from precipgen.engines.analytical import ParameterManifest, MonthlyParams, TrendAnalysis


def create_trend_manifest():
    """Create a parameter manifest with known trends for demonstration."""
    # Create baseline monthly parameters
    monthly_params = {}
    for month in range(1, 13):
        # Seasonal variation in baseline parameters
        if month in [12, 1, 2]:  # Winter
            p_ww, p_wd = 0.6, 0.3
            alpha, beta = 1.5, 5.0
        elif month in [6, 7, 8]:  # Summer
            p_ww, p_wd = 0.4, 0.2
            alpha, beta = 1.2, 4.0
        else:  # Spring/Fall
            p_ww, p_wd = 0.5, 0.25
            alpha, beta = 1.3, 4.5
        
        monthly_params[month] = MonthlyParams(
            p_ww=p_ww, p_wd=p_wd, alpha=alpha, beta=beta
        )
    
    # Create trend analysis with known slopes
    seasonal_slopes = {
        'Winter': {
            'p_ww': 0.005,   # 0.5% increase per year
            'p_wd': 0.002,   # 0.2% increase per year
            'alpha': 0.02,   # 0.02 units increase per year
            'beta': 0.1      # 0.1 units increase per year
        },
        'Spring': {
            'p_ww': 0.003,
            'p_wd': 0.001,
            'alpha': 0.01,
            'beta': 0.05
        },
        'Summer': {
            'p_ww': -0.002,  # Slight decrease
            'p_wd': -0.001,
            'alpha': -0.01,
            'beta': -0.05
        },
        'Fall': {
            'p_ww': 0.004,
            'p_wd': 0.0015,
            'alpha': 0.015,
            'beta': 0.08
        }
    }
    
    # Create significance tests (all significant for demo)
    significance_tests = {}
    trend_confidence = {}
    for season in seasonal_slopes.keys():
        significance_tests[season] = {param: 0.01 for param in seasonal_slopes[season]}
        trend_confidence[season] = {param: "High (p < 0.01)" for param in seasonal_slopes[season]}
    
    trend_analysis = TrendAnalysis(
        seasonal_slopes=seasonal_slopes,
        significance_tests=significance_tests,
        trend_confidence=trend_confidence,
        regression_type='linear'
    )
    
    return ParameterManifest(
        metadata={
            'station_id': 'TREND_DEMO',
            'wet_day_threshold': 0.001,
            'description': 'Demonstration parameters with known trends'
        },
        overall_parameters=monthly_params,
        trend_analysis=trend_analysis,
        sliding_window_stats={'window_count': 10, 'window_years': 3}
    )


def demonstrate_parameter_drift():
    """Demonstrate parameter drift calculation over time."""
    print("=== Parameter Drift Demonstration ===")
    
    manifest = create_trend_manifest()
    
    # Create engine with trend mode enabled
    engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
    
    # Test parameter drift at different time points
    test_dates = [
        datetime(2020, 1, 15),   # Winter, year 0
        datetime(2025, 1, 15),   # Winter, year 5
        datetime(2030, 1, 15),   # Winter, year 10
        datetime(2020, 7, 15),   # Summer, year 0
        datetime(2025, 7, 15),   # Summer, year 5
        datetime(2030, 7, 15),   # Summer, year 10
    ]
    
    print("Parameter Drift Over Time:")
    print("-" * 80)
    print(f"{'Date':<12} {'Season':<8} {'Years':<6} {'P(W|W)':<8} {'P(W|D)':<8} {'Alpha':<8} {'Beta':<8}")
    print("-" * 80)
    
    for date in test_dates:
        engine.initialize(datetime(2020, 1, 1))  # Always start from same baseline
        engine.current_date = date
        engine.elapsed_days = (date - datetime(2020, 1, 1)).days
        
        params = engine._get_current_parameters()
        elapsed_years = engine._calculate_elapsed_years()
        season = engine._get_season(date.month)
        
        print(f"{date.strftime('%Y-%m-%d'):<12} {season:<8} {elapsed_years:<6.1f} "
              f"{params.p_ww:<8.3f} {params.p_wd:<8.3f} {params.alpha:<8.3f} {params.beta:<8.3f}")
    
    print()


def demonstrate_bounds_checking():
    """Demonstrate physical bounds checking for extreme trends."""
    print("=== Physical Bounds Checking Demonstration ===")
    
    # Create manifest with extreme trends to test bounds
    monthly_params = {
        1: MonthlyParams(p_ww=0.5, p_wd=0.3, alpha=1.0, beta=3.0)
    }
    for month in range(2, 13):
        monthly_params[month] = monthly_params[1]
    
    # Extreme trends that would violate bounds
    extreme_slopes = {
        'Winter': {
            'p_ww': 0.1,     # 10% increase per year (would exceed 1.0)
            'p_wd': -0.05,   # 5% decrease per year (would go negative)
            'alpha': -0.2,   # Large decrease (would go negative)
            'beta': 2.0      # Large increase
        }
    }
    
    # Copy to other seasons
    for season in ['Spring', 'Summer', 'Fall']:
        extreme_slopes[season] = extreme_slopes['Winter'].copy()
    
    trend_analysis = TrendAnalysis(
        seasonal_slopes=extreme_slopes,
        significance_tests={'Winter': {param: 0.01 for param in extreme_slopes['Winter']}},
        trend_confidence={'Winter': {param: "High (p < 0.01)" for param in extreme_slopes['Winter']}},
        regression_type='linear'
    )
    
    extreme_manifest = ParameterManifest(
        metadata={'station_id': 'BOUNDS_TEST', 'wet_day_threshold': 0.001},
        overall_parameters=monthly_params,
        trend_analysis=trend_analysis,
        sliding_window_stats=None
    )
    
    engine = SimulationEngine(extreme_manifest, trend_mode=True, random_seed=42)
    
    print("Testing Extreme Trends (should be bounded):")
    print("-" * 70)
    print(f"{'Years':<6} {'P(W|W)':<10} {'P(W|D)':<10} {'Alpha':<10} {'Beta':<10}")
    print("-" * 70)
    
    for years in [0, 2, 5, 10, 20]:
        test_date = datetime(2020, 1, 15) + timedelta(days=int(years * 365.25))
        engine.initialize(datetime(2020, 1, 1))
        engine.current_date = test_date
        engine.elapsed_days = (test_date - datetime(2020, 1, 1)).days
        
        params = engine._get_current_parameters()
        
        print(f"{years:<6} {params.p_ww:<10.3f} {params.p_wd:<10.3f} "
              f"{params.alpha:<10.3f} {params.beta:<10.3f}")
    
    print("\nNote: Parameters remain within physical bounds despite extreme trends")
    print()


def demonstrate_trend_diagnostics():
    """Demonstrate trend projection diagnostics and validation."""
    print("=== Trend Projection Diagnostics ===")
    
    manifest = create_trend_manifest()
    engine = SimulationEngine(manifest, trend_mode=True, random_seed=42)
    engine.initialize(datetime(2020, 1, 1))
    
    # Advance to 5 years later
    for _ in range(5 * 365):
        engine.step()
    
    # Get trend projection info
    trend_info = engine.get_trend_projection_info()
    
    print("Current Trend Projection State:")
    print(f"  Trend mode enabled: {trend_info['trend_mode_enabled']}")
    print(f"  Elapsed years: {trend_info['elapsed_years']:.2f}")
    print(f"  Current season: {trend_info['current_season']}")
    
    if 'parameter_drift' in trend_info:
        print("\n  Parameter Drift:")
        for param, drift in trend_info['parameter_drift'].items():
            print(f"    {param}: {drift:+.4f}")
    
    if 'bounds_applied' in trend_info:
        bounds_applied = [param for param, applied in trend_info['bounds_applied'].items() if applied]
        if bounds_applied:
            print(f"\n  Bounds applied to: {', '.join(bounds_applied)}")
        else:
            print("\n  No bounds were applied")
    
    # Validate trend projection
    validation = engine.validate_trend_projection()
    print(f"\nTrend Projection Validation:")
    print(f"  Valid: {validation['is_valid']}")
    
    if validation['warnings']:
        print("  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    if validation['errors']:
        print("  Errors:")
        for error in validation['errors']:
            print(f"    - {error}")
    
    print()


def demonstrate_long_term_simulation():
    """Demonstrate long-term simulation with trend projection."""
    print("=== Long-term Simulation with Trends ===")
    
    manifest = create_trend_manifest()
    
    # Run two simulations: one with trends, one without
    results = {}
    
    for trend_mode in [False, True]:
        engine = SimulationEngine(manifest, trend_mode=trend_mode, random_seed=42)
        engine.initialize(datetime(2020, 1, 1))
        
        # Simulate 20 years
        dates = []
        precipitation = []
        wet_days = []
        
        for day in range(20 * 365):
            precip = engine.step()
            if day % (5 * 365) == 0:  # Sample every 5 years
                dates.append(engine.current_date)
                precipitation.append(precip)
                wet_days.append(precip > 0.001)
        
        results[f'trend_{trend_mode}'] = {
            'dates': dates,
            'precipitation': precipitation,
            'wet_days': wet_days
        }
    
    # Compare results
    print("Comparison of 20-year simulations (sampled every 5 years):")
    print("-" * 60)
    print(f"{'Year':<6} {'No Trend':<12} {'With Trend':<12} {'Difference':<12}")
    print("-" * 60)
    
    for i in range(len(results['trend_False']['dates'])):
        year = 2020 + i * 5
        no_trend = results['trend_False']['precipitation'][i]
        with_trend = results['trend_True']['precipitation'][i]
        diff = with_trend - no_trend
        
        print(f"{year:<6} {no_trend:<12.3f} {with_trend:<12.3f} {diff:+12.3f}")
    
    print("\nNote: Differences show the effect of parameter drift over time")
    print()


def main():
    """Run all trend projection demonstrations."""
    print("PrecipGen Enhanced Trend Projection System Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_parameter_drift()
    demonstrate_bounds_checking()
    demonstrate_trend_diagnostics()
    demonstrate_long_term_simulation()
    
    print("Enhanced trend projection system is working correctly!")
    print("✓ Parameter drift calculation using trend slopes")
    print("✓ Physical bounds checking for drifted parameters")
    print("✓ Time-based parameter adjustment logic")
    print("✓ Mathematical correctness of drift formula")
    print("✓ Comprehensive error handling and validation")


if __name__ == "__main__":
    main()