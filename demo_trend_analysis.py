#!/usr/bin/env python3
"""
Demonstration of enhanced trend analysis functionality.

This script shows how to use the improved trend analysis system with
both linear and polynomial regression options.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from precipgen.engines import AnalyticalEngine


def create_sample_data_with_trend():
    """Create sample precipitation data with a known trend."""
    # Create 10 years of daily data
    dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Create seasonal variation
    day_of_year = dates.dayofyear
    seasonal_wet_prob = 0.25 + 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Add increasing trend over time
    years = dates.year.values
    year_progress = (years - years.min()) / (years.max() - years.min())
    trend_wet_prob = seasonal_wet_prob + 0.1 * year_progress  # 10% increase over 10 years
    trend_wet_prob = np.clip(trend_wet_prob, 0.05, 0.95)
    
    # Generate wet/dry days
    wet_days = np.random.random(len(dates)) < trend_wet_prob
    
    # Generate precipitation amounts with slight trend in intensity
    base_alpha = 1.5
    base_beta = 5.0
    trend_alpha = base_alpha + 0.3 * year_progress  # Slight increase in shape parameter
    
    precip_amounts = np.where(
        wet_days,
        np.random.gamma(trend_alpha, base_beta),
        0.0
    )
    
    return pd.Series(precip_amounts, index=dates)


def demonstrate_trend_analysis():
    """Demonstrate the enhanced trend analysis functionality."""
    print("PrecipGen Enhanced Trend Analysis Demonstration")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample precipitation data with known trends...")
    data = create_sample_data_with_trend()
    
    print(f"Data period: {data.index.min()} to {data.index.max()}")
    print(f"Total data points: {len(data)}")
    print(f"Wet days: {(data > 0.001).sum()} ({(data > 0.001).mean()*100:.1f}%)")
    print()
    
    # Initialize analytical engine
    engine = AnalyticalEngine(data, wet_day_threshold=0.001)
    engine.initialize()
    
    # Perform comprehensive trend analysis with linear regression
    print("Performing comprehensive trend analysis (linear regression)...")
    linear_trends = engine.perform_comprehensive_trend_analysis(
        window_years=3, 
        regression_type='linear'
    )
    
    print("Linear Trend Analysis Results:")
    print("-" * 30)
    for season, slopes in linear_trends.seasonal_slopes.items():
        print(f"\n{season}:")
        for param, slope in slopes.items():
            p_value = linear_trends.significance_tests[season][param]
            confidence = linear_trends.trend_confidence[season][param]
            print(f"  {param}: {slope:+.6f}/year (p={p_value:.3f}, {confidence})")
    
    # Perform trend analysis with polynomial regression
    print("\n" + "=" * 50)
    print("Performing trend analysis with polynomial regression...")
    
    # Reset engine to clear previous analysis
    engine.reset()
    window_analysis = engine.perform_sliding_window_analysis(window_years=3)
    poly_trends = engine.extract_trends(window_analysis, regression_type='polynomial')
    
    print("Polynomial Trend Analysis Results:")
    print("-" * 30)
    for season, slopes in poly_trends.seasonal_slopes.items():
        print(f"\n{season}:")
        for param, slope in slopes.items():
            p_value = poly_trends.significance_tests[season][param]
            confidence = poly_trends.trend_confidence[season][param]
            print(f"  {param}: {slope:+.6f}/year (p={p_value:.3f}, {confidence})")
    
    # Show validation results
    print("\n" + "=" * 50)
    print("Trend Slope Validation Results:")
    print("-" * 30)
    
    for regression_type, trends in [('Linear', linear_trends), ('Polynomial', poly_trends)]:
        print(f"\n{regression_type} Regression Validation:")
        if trends.validation_results:
            for season, season_validation in trends.validation_results.items():
                invalid_params = [param for param, valid in season_validation.items() if not valid]
                if invalid_params:
                    print(f"  {season}: Invalid slopes for {invalid_params}")
                else:
                    print(f"  {season}: All slopes within reasonable bounds")
        else:
            print("  No validation results available")
    
    # Generate parameter manifest
    print("\n" + "=" * 50)
    print("Parameter Manifest Summary:")
    print("-" * 30)
    
    manifest = engine.generate_parameter_manifest()
    print(f"Analysis date: {manifest.metadata['analysis_date']}")
    print(f"Data completeness: {manifest.metadata['data_completeness']:.3f}")
    print(f"Wet day threshold: {manifest.metadata['wet_day_threshold']} inches")
    
    if manifest.sliding_window_stats:
        print(f"Window analysis: {manifest.sliding_window_stats['window_count']} windows")
        print(f"Window size: {manifest.sliding_window_stats['window_years']} years")
        print(f"Regression type: {manifest.sliding_window_stats['regression_type']}")
        print(f"Validation performed: {manifest.sliding_window_stats['trend_validation_performed']}")
    
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_trend_analysis()