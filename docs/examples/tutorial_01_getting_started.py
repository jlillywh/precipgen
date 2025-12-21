#!/usr/bin/env python3
"""
Tutorial 1: Getting Started with PrecipGen

This tutorial provides a complete introduction to PrecipGen for new users.
We'll walk through every step from installation to generating your first
synthetic precipitation data.

Author: PrecipGen Development Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import PrecipGen - if this fails, install with: pip install precipgen
try:
    import precipgen as pg
    print("✓ PrecipGen imported successfully")
except ImportError:
    print("✗ PrecipGen not found. Install with: pip install precipgen")
    exit(1)

def tutorial_step_1():
    """Step 1: Understanding precipitation data structure"""
    
    print("\n" + "="*60)
    print("STEP 1: Understanding Precipitation Data")
    print("="*60)
    
    print("\nPrecipGen works with daily precipitation time series.")
    print("The expected format is a pandas Series with:")
    print("- DatetimeIndex for dates")
    print("- Float values for daily precipitation amounts")
    print("- Units typically in mm or inches")
    
    # Create a simple example
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    precip_values = [0.0, 2.5, 0.0, 1.2, 0.0, 0.0, 5.1, 0.8, 0.0, 0.0]
    
    precip_series = pd.Series(precip_values, index=dates, name='precipitation')
    
    print("\nExample precipitation data:")
    print(precip_series)
    
    print(f"\nData type: {type(precip_series)}")
    print(f"Index type: {type(precip_series.index)}")
    print(f"Value type: {precip_series.dtype}")
    
    return precip_series

def tutorial_step_2():
    """Step 2: Creating realistic test data"""
    
    print("\n" + "="*60)
    print("STEP 2: Creating Realistic Test Data")
    print("="*60)
    
    print("\nFor this tutorial, we'll create synthetic precipitation data")
    print("that mimics real climate patterns with seasonal variation.")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create 10 years of daily data
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2019, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    print(f"\nGenerating {len(dates)} days of data ({start_date.year}-{end_date.year})")
    
    precipitation = []
    is_wet_yesterday = False
    
    for date in dates:
        month = date.month
        
        # Simple seasonal parameters
        if month in [12, 1, 2]:  # Winter - more persistent, moderate amounts
            p_ww, p_wd = 0.7, 0.3  # Higher persistence
            alpha, beta = 1.0, 6.0  # Moderate amounts
        elif month in [6, 7, 8]:  # Summer - less persistent, variable amounts
            p_ww, p_wd = 0.4, 0.2  # Lower persistence
            alpha, beta = 1.5, 4.0  # More variable amounts
        else:  # Spring/Fall - intermediate
            p_ww, p_wd = 0.5, 0.25
            alpha, beta = 1.2, 5.0
        
        # Markov chain for wet/dry state
        if is_wet_yesterday:
            prob_wet_today = p_ww
        else:
            prob_wet_today = p_wd
        
        is_wet_today = np.random.random() < prob_wet_today
        
        # Generate precipitation amount
        if is_wet_today:
            # Sample from Gamma distribution
            precip = np.random.gamma(alpha, beta)
        else:
            precip = 0.0
        
        precipitation.append(precip)
        is_wet_yesterday = is_wet_today
    
    # Create the time series
    precip_data = pd.Series(precipitation, index=dates, name='precipitation')
    
    # Display basic statistics
    print(f"\nGenerated data statistics:")
    print(f"Total days: {len(precip_data)}")
    print(f"Mean daily precipitation: {precip_data.mean():.2f} mm")
    print(f"Maximum daily precipitation: {precip_data.max():.2f} mm")
    print(f"Wet days (>0.1mm): {(precip_data > 0.1).sum()} ({(precip_data > 0.1).mean():.1%})")
    print(f"Annual total: {precip_data.sum():.0f} mm")
    
    return precip_data

def tutorial_step_3(precip_data):
    """Step 3: Basic configuration"""
    
    print("\n" + "="*60)
    print("STEP 3: Configuring PrecipGen")
    print("="*60)
    
    print("\nPrecipGen uses a configuration system to manage settings.")
    print("Let's create a basic configuration:")
    
    # Create configuration
    config = pg.PrecipGenConfig({
        'analysis': {
            'wet_day_threshold': 0.1,  # mm - days with less are considered dry
            'sliding_window_years': 15,  # For trend analysis
            'trend_significance_level': 0.05  # p-value threshold
        },
        'quality': {
            'min_completeness': 0.8,  # Require 80% data completeness
            'physical_bounds': {
                'min_precip': 0.0,
                'max_precip': 200.0  # mm - reasonable daily maximum
            }
        }
    })
    
    print("\nConfiguration created with:")
    print(f"- Wet day threshold: {config.analysis.wet_day_threshold} mm")
    print(f"- Minimum data completeness: {config.quality.min_completeness:.0%}")
    print(f"- Maximum reasonable daily precipitation: {config.quality.physical_bounds['max_precip']} mm")
    
    # Validate configuration
    print("\nValidating configuration...")
    errors = config.validate()
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Configuration is valid")
    
    return config

def tutorial_step_4(precip_data, config):
    """Step 4: Data quality assessment"""
    
    print("\n" + "="*60)
    print("STEP 4: Assessing Data Quality")
    print("="*60)
    
    print("\nBefore analysis, we should check data quality.")
    print("PrecipGen provides tools to assess completeness and validity.")
    
    # Create data validator
    validator = pg.DataValidator(config.quality)
    
    # Assess overall data quality
    print("\nPerforming comprehensive quality assessment...")
    quality_report = validator.assess_data_quality(precip_data)
    
    print(f"\nQuality Assessment Results:")
    print(f"Data completeness: {quality_report.completeness:.1%}")
    print(f"Quality score: {quality_report.quality_score:.3f}")
    print(f"Values outside physical bounds: {quality_report.out_of_bounds_count}")
    print(f"Recommendation: {quality_report.recommendation}")
    
    # Detailed checks
    print("\nDetailed quality checks:")
    
    # Check completeness
    completeness_result = validator.validate_completeness(precip_data)
    print(f"✓ Completeness check: {completeness_result.status}")
    
    # Check physical bounds
    bounds_result = validator.validate_physical_bounds(precip_data)
    print(f"✓ Physical bounds check: {bounds_result.status}")
    
    if quality_report.recommendation == 'ACCEPT':
        print("\n✓ Data quality is sufficient for analysis")
    else:
        print("\n⚠ Data quality may be insufficient - proceed with caution")
    
    return quality_report

def tutorial_step_5(precip_data):
    """Step 5: Parameter estimation"""
    
    print("\n" + "="*60)
    print("STEP 5: Estimating Precipitation Parameters")
    print("="*60)
    
    print("\nNow we'll analyze the precipitation data to extract")
    print("the statistical parameters needed for simulation.")
    
    # Create analytical engine
    print("\nCreating analytical engine...")
    engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.1)
    
    # Calculate monthly parameters
    print("Calculating monthly parameters...")
    monthly_params = engine.calculate_monthly_parameters()
    
    print(f"\nEstimated parameters for {len(monthly_params)} months:")
    print("\nMonth  P(W|W)  P(W|D)  Alpha   Beta    Wet%")
    print("-" * 45)
    
    for month, params in monthly_params.items():
        # Calculate wet day frequency for this month
        month_data = precip_data[precip_data.index.month == month]
        wet_freq = (month_data > 0.1).mean()
        
        print(f"{month:2d}     {params.p_ww:.3f}   {params.p_wd:.3f}   {params.alpha:.2f}   {params.beta:.2f}   {wet_freq:.1%}")
    
    print("\nParameter explanation:")
    print("- P(W|W): Probability of wet day following wet day")
    print("- P(W|D): Probability of wet day following dry day")
    print("- Alpha: Gamma distribution shape parameter")
    print("- Beta: Gamma distribution scale parameter")
    print("- Wet%: Observed wet day frequency")
    
    return engine, monthly_params

def tutorial_step_6(engine):
    """Step 6: Generate parameter manifest"""
    
    print("\n" + "="*60)
    print("STEP 6: Creating Parameter Manifest")
    print("="*60)
    
    print("\nThe parameter manifest contains all information needed")
    print("for synthetic precipitation generation.")
    
    # Generate manifest
    print("Generating parameter manifest...")
    manifest = engine.generate_parameter_manifest()
    
    print(f"\nManifest created with:")
    print(f"Station ID: {manifest['metadata']['station_id']}")
    print(f"Data period: {manifest['metadata']['data_period'][0]} to {manifest['metadata']['data_period'][1]}")
    print(f"Wet day threshold: {manifest['metadata']['wet_day_threshold']} mm")
    print(f"Data completeness: {manifest['metadata']['data_completeness']:.1%}")
    
    print(f"\nManifest contains:")
    print(f"- Overall parameters: {len(manifest['overall_parameters']['monthly'])} months")
    print(f"- Trend analysis: {'Yes' if 'trend_analysis' in manifest else 'No'}")
    
    # Save manifest to file
    import json
    with open('tutorial_parameters.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print("\n✓ Parameter manifest saved to 'tutorial_parameters.json'")
    
    return manifest

def tutorial_step_7(manifest):
    """Step 7: Generate synthetic precipitation"""
    
    print("\n" + "="*60)
    print("STEP 7: Generating Synthetic Precipitation")
    print("="*60)
    
    print("\nNow we'll use the estimated parameters to generate")
    print("synthetic precipitation data.")
    
    # Create simulation engine
    print("Creating simulation engine...")
    sim = pg.SimulationEngine(
        parameters=manifest,
        trend_mode=False,  # Start with stationary simulation
        random_seed=123    # For reproducible results
    )
    
    # Initialize simulation
    start_date = datetime(2025, 1, 1)
    print(f"Initializing simulation starting {start_date.date()}...")
    sim.initialize(start_date, initial_wet_state=False)
    
    # Generate one year of synthetic data
    print("Generating 365 days of synthetic precipitation...")
    
    synthetic_data = []
    dates = []
    current_date = start_date
    
    for day in range(365):
        # Generate precipitation for this day
        daily_precip = sim.step()
        synthetic_data.append(daily_precip)
        dates.append(current_date)
        current_date += timedelta(days=1)
        
        # Show progress every 50 days
        if (day + 1) % 50 == 0:
            print(f"  Generated {day + 1} days...")
    
    # Create synthetic time series
    synthetic_series = pd.Series(synthetic_data, index=dates, name='synthetic_precipitation')
    
    print(f"\n✓ Generated {len(synthetic_data)} days of synthetic precipitation")
    
    # Display basic statistics
    print(f"\nSynthetic data statistics:")
    print(f"Mean daily precipitation: {synthetic_series.mean():.2f} mm")
    print(f"Maximum daily precipitation: {synthetic_series.max():.2f} mm")
    print(f"Wet days (>0.1mm): {(synthetic_series > 0.1).sum()} ({(synthetic_series > 0.1).mean():.1%})")
    print(f"Annual total: {synthetic_series.sum():.0f} mm")
    
    return synthetic_series

def tutorial_step_8(precip_data, synthetic_series):
    """Step 8: Compare historical vs synthetic"""
    
    print("\n" + "="*60)
    print("STEP 8: Comparing Historical vs Synthetic Data")
    print("="*60)
    
    print("\nLet's compare the synthetic data with the original historical data")
    print("to see how well our model reproduces the statistical properties.")
    
    # Calculate comparison statistics
    print("\nOverall Statistics Comparison:")
    print("Metric                    Historical    Synthetic    Difference")
    print("-" * 60)
    
    # Mean precipitation
    hist_mean = precip_data.mean()
    synth_mean = synthetic_series.mean()
    print(f"Mean daily precip (mm)    {hist_mean:8.2f}      {synth_mean:8.2f}      {synth_mean-hist_mean:+7.2f}")
    
    # Wet day frequency
    hist_wet_freq = (precip_data > 0.1).mean()
    synth_wet_freq = (synthetic_series > 0.1).mean()
    print(f"Wet day frequency         {hist_wet_freq:8.1%}       {synth_wet_freq:8.1%}       {synth_wet_freq-hist_wet_freq:+7.1%}")
    
    # Annual total (scaled for historical data)
    hist_annual = precip_data.sum() / (len(precip_data) / 365.25)
    synth_annual = synthetic_series.sum()
    print(f"Annual total (mm)         {hist_annual:8.0f}       {synth_annual:8.0f}       {synth_annual-hist_annual:+7.0f}")
    
    # Maximum daily amount
    hist_max = precip_data.max()
    synth_max = synthetic_series.max()
    print(f"Maximum daily (mm)        {hist_max:8.2f}      {synth_max:8.2f}      {synth_max-hist_max:+7.2f}")
    
    # Monthly comparison
    print("\nMonthly Statistics Comparison:")
    print("Month  Historical Mean  Synthetic Mean  Historical Wet%  Synthetic Wet%")
    print("-" * 70)
    
    for month in range(1, 13):
        # Historical monthly stats
        hist_month = precip_data[precip_data.index.month == month]
        hist_month_mean = hist_month.mean()
        hist_month_wet = (hist_month > 0.1).mean()
        
        # Synthetic monthly stats (if we have data for that month)
        synth_month = synthetic_series[synthetic_series.index.month == month]
        if len(synth_month) > 0:
            synth_month_mean = synth_month.mean()
            synth_month_wet = (synth_month > 0.1).mean()
        else:
            synth_month_mean = 0.0
            synth_month_wet = 0.0
        
        print(f"{month:2d}     {hist_month_mean:10.2f}     {synth_month_mean:10.2f}        {hist_month_wet:8.1%}        {synth_month_wet:8.1%}")
    
    print("\nInterpretation:")
    print("- Small differences are expected due to random variation")
    print("- Large systematic differences may indicate model issues")
    print("- Monthly patterns should be reasonably preserved")

def tutorial_step_9(synthetic_series):
    """Step 9: Save and export results"""
    
    print("\n" + "="*60)
    print("STEP 9: Saving Results")
    print("="*60)
    
    print("\nLet's save our synthetic precipitation data for future use.")
    
    # Create DataFrame for export
    synthetic_df = pd.DataFrame({
        'date': synthetic_series.index,
        'precipitation_mm': synthetic_series.values
    })
    
    # Save to CSV
    csv_filename = 'tutorial_synthetic_precipitation.csv'
    synthetic_df.to_csv(csv_filename, index=False)
    print(f"✓ Synthetic data saved to '{csv_filename}'")
    
    # Save summary statistics
    summary_stats = {
        'generation_info': {
            'generated_date': datetime.now().isoformat(),
            'random_seed': 123,
            'simulation_mode': 'stationary',
            'total_days': len(synthetic_series)
        },
        'statistics': {
            'mean_daily_mm': float(synthetic_series.mean()),
            'max_daily_mm': float(synthetic_series.max()),
            'annual_total_mm': float(synthetic_series.sum()),
            'wet_day_frequency': float((synthetic_series > 0.1).mean()),
            'wet_day_count': int((synthetic_series > 0.1).sum())
        }
    }
    
    import json
    with open('tutorial_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print("✓ Summary statistics saved to 'tutorial_summary.json'")
    
    print(f"\nFiles created:")
    print(f"- tutorial_parameters.json (parameter manifest)")
    print(f"- {csv_filename} (synthetic precipitation data)")
    print(f"- tutorial_summary.json (summary statistics)")

def tutorial_step_10():
    """Step 10: Next steps and advanced features"""
    
    print("\n" + "="*60)
    print("STEP 10: Next Steps and Advanced Features")
    print("="*60)
    
    print("\nCongratulations! You've completed the basic PrecipGen tutorial.")
    print("You've learned how to:")
    print("✓ Configure PrecipGen")
    print("✓ Assess data quality")
    print("✓ Estimate precipitation parameters")
    print("✓ Generate synthetic precipitation")
    print("✓ Compare results with historical data")
    
    print("\nAdvanced features to explore next:")
    
    print("\n1. TREND ANALYSIS:")
    print("   - Use sliding window analysis to detect parameter changes over time")
    print("   - Generate non-stationary precipitation with climate trends")
    print("   - Example: sim = pg.SimulationEngine(manifest, trend_mode=True)")
    
    print("\n2. BOOTSTRAP RESAMPLING:")
    print("   - Generate precipitation by resampling historical years")
    print("   - Useful when you want to preserve exact historical patterns")
    print("   - Example: bootstrap = pg.BootstrapEngine(historical_data, mode='random')")
    
    print("\n3. GHCN DATA PROCESSING:")
    print("   - Work directly with GHCN Daily (.dly) format files")
    print("   - Automatic parsing and quality flag handling")
    print("   - Example: parser = pg.GHCNParser(); data = parser.parse_dly_file('station.dly')")
    
    print("\n4. BATCH PROCESSING:")
    print("   - Process multiple weather stations simultaneously")
    print("   - Generate ensemble simulations with different random seeds")
    print("   - Useful for uncertainty quantification")
    
    print("\n5. EXTERNAL MODEL INTEGRATION:")
    print("   - Integrate PrecipGen with hydrological or agricultural models")
    print("   - Use StandardizedAPI for consistent data exchange")
    print("   - Synchronize with external simulation clocks")
    
    print("\nRecommended next tutorials:")
    print("- tutorial_02_advanced_analysis.py (trend analysis and non-stationary simulation)")
    print("- tutorial_03_custom_workflows.py (building custom analysis pipelines)")
    
    print("\nDocumentation and examples:")
    print("- docs/user_guide.md (comprehensive usage guide)")
    print("- docs/api_reference.md (complete API documentation)")
    print("- docs/examples/ (working code examples)")
    
    print("\nHappy weather generation! 🌧️")

def main():
    """Run the complete tutorial"""
    
    print("PrecipGen Tutorial 1: Getting Started")
    print("====================================")
    print("\nThis tutorial will guide you through the basic workflow of PrecipGen.")
    print("We'll generate synthetic precipitation data step by step.")
    
    try:
        # Run all tutorial steps
        tutorial_step_1()
        precip_data = tutorial_step_2()
        config = tutorial_step_3(precip_data)
        tutorial_step_4(precip_data, config)
        engine, monthly_params = tutorial_step_5(precip_data)
        manifest = tutorial_step_6(engine)
        synthetic_series = tutorial_step_7(manifest)
        tutorial_step_8(precip_data, synthetic_series)
        tutorial_step_9(synthetic_series)
        tutorial_step_10()
        
        print("\n" + "="*60)
        print("TUTORIAL COMPLETE!")
        print("="*60)
        print("\nYou have successfully:")
        print("✓ Generated realistic test precipitation data")
        print("✓ Configured and validated PrecipGen settings")
        print("✓ Estimated statistical parameters from historical data")
        print("✓ Generated synthetic precipitation using WGEN algorithm")
        print("✓ Compared synthetic vs historical statistics")
        print("✓ Saved results for future use")
        
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("- Ensure PrecipGen is properly installed: pip install precipgen")
        print("- Check that all required dependencies are available")
        print("- Verify you have write permissions in the current directory")
        raise

if __name__ == "__main__":
    main()