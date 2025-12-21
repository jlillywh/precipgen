#!/usr/bin/env python3
"""
Demonstration of the AnalyticalEngine functionality.

This script shows how to use the AnalyticalEngine to:
1. Analyze precipitation data
2. Calculate monthly parameters
3. Perform sliding window analysis
4. Extract trends
5. Generate parameter manifests
6. Export results to JSON files
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from precipgen.engines import AnalyticalEngine


def create_sample_data():
    """Create sample precipitation data with seasonal patterns and trends."""
    print("Creating sample precipitation data...")
    
    # Create 10 years of daily data
    dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate seasonal pattern
    day_of_year = dates.dayofyear.values
    seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Add long-term trend
    years = dates.year.values
    year_progress = (years - years.min()) / (years.max() - years.min())
    trend_factor = 1 + 0.3 * year_progress  # 30% increase over 10 years
    
    # Generate wet/dry pattern
    base_wet_prob = 0.25
    wet_prob = base_wet_prob * seasonal_factor * trend_factor / (seasonal_factor.mean() * 1.15)
    wet_days = np.random.random(len(dates)) < wet_prob
    
    # Generate precipitation amounts for wet days
    precip_amounts = np.where(
        wet_days,
        np.random.gamma(1.5, 5.0, len(dates)),  # Gamma distribution for wet day amounts
        0.0
    )
    
    data = pd.Series(precip_amounts, index=dates)
    
    print(f"Created {len(data)} days of data from {data.index.min()} to {data.index.max()}")
    print(f"Wet day fraction: {(data > 0).sum() / len(data):.3f}")
    print(f"Mean precipitation on wet days: {data[data > 0].mean():.2f} mm")
    
    return data


def demonstrate_basic_analysis():
    """Demonstrate basic parameter calculation."""
    print("\n" + "="*60)
    print("BASIC PARAMETER ANALYSIS")
    print("="*60)
    
    # Create sample data
    data = create_sample_data()
    
    # Initialize analytical engine
    engine = AnalyticalEngine(data, wet_day_threshold=0.001)
    engine.initialize()
    
    print(f"\nEngine state: {engine.get_state()}")
    
    # Calculate monthly parameters
    print("\nCalculating monthly parameters...")
    monthly_params = engine.calculate_monthly_parameters()
    
    print("\nMonthly Parameters:")
    print("-" * 50)
    for month, params in monthly_params.items():
        month_name = pd.Timestamp(2000, month, 1).strftime('%B')
        print(f"{month_name:>10}: P(W|W)={params.p_ww:.3f}, P(W|D)={params.p_wd:.3f}, "
              f"α={params.alpha:.2f}, β={params.beta:.2f}")
    
    return engine


def demonstrate_sliding_window_analysis(engine):
    """Demonstrate sliding window analysis."""
    print("\n" + "="*60)
    print("SLIDING WINDOW ANALYSIS")
    print("="*60)
    
    # Perform sliding window analysis
    print("Performing sliding window analysis (3-year windows)...")
    window_analysis = engine.perform_sliding_window_analysis(window_years=3)
    
    print(f"\nGenerated {len(window_analysis.window_parameters)} windows")
    
    # Show first few windows
    print("\nFirst 3 windows:")
    print("-" * 50)
    for i, (window_id, window_params) in enumerate(list(window_analysis.window_parameters.items())[:3]):
        start_date, end_date = window_analysis.window_dates[window_id]
        print(f"{window_id}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Months with data: {len(window_params)}")
        
        # Show January parameters if available
        if 1 in window_params:
            jan_params = window_params[1]
            print(f"  January: P(W|W)={jan_params.p_ww:.3f}, P(W|D)={jan_params.p_wd:.3f}")
    
    return window_analysis


def demonstrate_trend_analysis(engine, window_analysis):
    """Demonstrate trend extraction."""
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    # Extract trends
    print("Extracting trends using linear regression...")
    trend_analysis = engine.extract_trends(window_analysis, regression_type='linear')
    
    print(f"\nTrend analysis completed using {trend_analysis.regression_type} regression")
    
    # Show seasonal trends
    print("\nSeasonal Trend Slopes (per year):")
    print("-" * 50)
    for season, slopes in trend_analysis.seasonal_slopes.items():
        print(f"\n{season}:")
        for param, slope in slopes.items():
            significance = trend_analysis.significance_tests[season][param]
            confidence = trend_analysis.trend_confidence[season][param]
            print(f"  {param:>5}: {slope:+.6f}/year (p={significance:.3f}, {confidence})")
    
    # Show validation results
    if trend_analysis.validation_results:
        print("\nTrend Slope Validation:")
        print("-" * 30)
        invalid_slopes = []
        for season, season_results in trend_analysis.validation_results.items():
            for param, is_valid in season_results.items():
                if not is_valid:
                    slope_value = trend_analysis.seasonal_slopes[season][param]
                    invalid_slopes.append(f"{season} {param}: {slope_value:.6f}")
        
        if invalid_slopes:
            print("Invalid slopes detected:")
            for slope_info in invalid_slopes:
                print(f"  - {slope_info}")
        else:
            print("All trend slopes are within reasonable bounds")
    
    return trend_analysis


def demonstrate_parameter_manifest(engine):
    """Demonstrate parameter manifest generation."""
    print("\n" + "="*60)
    print("PARAMETER MANIFEST GENERATION")
    print("="*60)
    
    # Generate parameter manifest
    print("Generating parameter manifest...")
    manifest = engine.generate_parameter_manifest()
    
    print("\nManifest Contents:")
    print("-" * 30)
    print(f"Metadata keys: {list(manifest.metadata.keys())}")
    print(f"Overall parameters: {len(manifest.overall_parameters)} months")
    print(f"Trend analysis included: {manifest.trend_analysis is not None}")
    print(f"Sliding window stats included: {manifest.sliding_window_stats is not None}")
    
    # Show some metadata
    print(f"\nData period: {manifest.metadata['data_period'][0]} to {manifest.metadata['data_period'][1]}")
    print(f"Data completeness: {manifest.metadata['data_completeness']:.3f}")
    print(f"Wet day threshold: {manifest.metadata['wet_day_threshold']} inches")
    
    if manifest.sliding_window_stats:
        print(f"Window count: {manifest.sliding_window_stats['window_count']}")
        print(f"Window size: {manifest.sliding_window_stats['window_years']} years")
    
    return manifest


def demonstrate_json_export(engine, manifest):
    """Demonstrate JSON export functionality."""
    print("\n" + "="*60)
    print("JSON EXPORT")
    print("="*60)
    
    # Convert to JSON string
    print("Converting manifest to JSON...")
    json_str = manifest.to_json(indent=2)
    print(f"JSON string length: {len(json_str)} characters")
    
    # Show first few lines of JSON
    json_lines = json_str.split('\n')
    print(f"\nFirst 10 lines of JSON:")
    print("-" * 30)
    for line in json_lines[:10]:
        print(line)
    print("...")
    
    # Export to files
    print("\nExporting results to files...")
    output_files = engine.export_results('output', include_json=True, include_report=True)
    
    print("Files created:")
    for file_type, filepath in output_files.items():
        print(f"  {file_type}: {filepath}")
    
    # Load and show analysis report summary
    with open(output_files['analysis_report'], 'r') as f:
        report = json.load(f)
    
    print(f"\nAnalysis Report Summary:")
    print(f"  Status: {report['analysis_status']}")
    print(f"  Errors: {len(report['errors'])}")
    print(f"  Warnings: {len(report['warnings'])}")
    print(f"  Data completeness: {report['data_quality']['data_completeness']:.3f}")
    
    if report['warnings']:
        print("  Warnings:")
        for warning in report['warnings']:
            print(f"    - {warning}")


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Test with empty data
    print("Testing with empty data...")
    empty_data = pd.Series([], dtype=float)
    engine = AnalyticalEngine(empty_data, wet_day_threshold=0.001)
    engine.initialize()
    
    report = engine.generate_analysis_report()
    print(f"Empty data analysis status: {report['analysis_status']}")
    if report['errors']:
        print("Errors detected:")
        for error in report['errors']:
            print(f"  - {error}")
    
    # Test with sparse data (low wet day fraction)
    print("\nTesting with sparse data...")
    dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
    sparse_data = pd.Series(np.where(np.random.random(len(dates)) < 0.01, 5.0, 0.0), index=dates)
    
    engine = AnalyticalEngine(sparse_data, wet_day_threshold=0.001)
    engine.initialize()
    engine.calculate_monthly_parameters()
    
    report = engine.generate_analysis_report()
    print(f"Sparse data analysis status: {report['analysis_status']}")
    if report['warnings']:
        print("Warnings detected:")
        for warning in report['warnings']:
            print(f"  - {warning}")


def main():
    """Run complete demonstration."""
    print("PRECIPGEN ANALYTICAL ENGINE DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the complete functionality of the AnalyticalEngine")
    print("including parameter calculation, trend analysis, and JSON export.")
    
    try:
        # Basic analysis
        engine = demonstrate_basic_analysis()
        
        # Sliding window analysis
        window_analysis = demonstrate_sliding_window_analysis(engine)
        
        # Trend analysis
        trend_analysis = demonstrate_trend_analysis(engine, window_analysis)
        
        # Parameter manifest
        manifest = demonstrate_parameter_manifest(engine)
        
        # JSON export
        demonstrate_json_export(engine, manifest)
        
        # Error handling
        demonstrate_error_handling()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Check the 'output' directory for exported JSON files:")
        print("  - parameter_manifest.json: Complete parameter manifest")
        print("  - analysis_report.json: Analysis report with quality metrics")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        raise


if __name__ == "__main__":
    main()