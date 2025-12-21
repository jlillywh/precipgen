#!/usr/bin/env python3
"""
Basic Analysis Example

This example demonstrates the fundamental workflow of PrecipGen:
1. Generate or load precipitation data
2. Configure the library
3. Perform parameter estimation
4. Generate synthetic precipitation

Author: PrecipGen Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import precipgen as pg

def generate_sample_data(years=30, seed=42):
    """
    Generate synthetic precipitation data for demonstration.
    
    This creates realistic-looking precipitation data with seasonal patterns
    and wet/dry spell characteristics similar to real climate data.
    """
    np.random.seed(seed)
    
    # Create date range
    start_date = datetime(1990, 1, 1)
    end_date = datetime(1990 + years, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    precipitation = []
    is_wet_yesterday = False
    
    for date in dates:
        month = date.month
        
        # Seasonal parameters (higher wet probability in winter)
        if month in [12, 1, 2]:  # Winter
            p_ww, p_wd = 0.6, 0.3
            alpha, beta = 1.2, 8.0
        elif month in [6, 7, 8]:  # Summer  
            p_ww, p_wd = 0.4, 0.15
            alpha, beta = 1.5, 5.0
        else:  # Spring/Fall
            p_ww, p_wd = 0.5, 0.25
            alpha, beta = 1.3, 6.5
        
        # Determine wet/dry state
        if is_wet_yesterday:
            prob_wet = p_ww
        else:
            prob_wet = p_wd
            
        is_wet_today = np.random.random() < prob_wet
        
        # Generate precipitation amount
        if is_wet_today:
            precip = np.random.gamma(alpha, beta)
        else:
            precip = 0.0
            
        precipitation.append(precip)
        is_wet_yesterday = is_wet_today
    
    # Create time series
    precip_series = pd.Series(precipitation, index=dates, name='precipitation')
    return precip_series

def main():
    """Main analysis workflow demonstration."""
    
    print("PrecipGen Basic Analysis Example")
    print("=" * 40)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample precipitation data...")
    precip_data = generate_sample_data(years=30, seed=42)
    
    print(f"   Data period: {precip_data.index[0].date()} to {precip_data.index[-1].date()}")
    print(f"   Total days: {len(precip_data)}")
    print(f"   Mean daily precipitation: {precip_data.mean():.2f} mm")
    print(f"   Wet day frequency: {(precip_data > 0.001).mean():.1%}")
    
    # Step 2: Configure PrecipGen
    print("\n2. Configuring PrecipGen...")
    config = pg.PrecipGenConfig({
        'analysis': {
            'wet_day_threshold': 0.001,  # mm
            'sliding_window_years': 20,
            'trend_significance_level': 0.05
        },
        'quality': {
            'min_completeness': 0.8,
            'physical_bounds': {
                'min_precip': 0.0,
                'max_precip': 500.0
            }
        }
    })
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("   Configuration errors:")
        for error in errors:
            print(f"     - {error}")
        return
    else:
        print("   Configuration validated successfully")
    
    # Step 3: Validate data quality
    print("\n3. Assessing data quality...")
    validator = pg.DataValidator(config.quality)
    quality_report = validator.assess_data_quality(precip_data)
    
    print(f"   Data completeness: {quality_report.completeness:.1%}")
    print(f"   Quality score: {quality_report.quality_score:.3f}")
    print(f"   Out of bounds values: {quality_report.out_of_bounds_count}")
    print(f"   Recommendation: {quality_report.recommendation}")
    
    if quality_report.recommendation != 'ACCEPT':
        print("   Warning: Data quality may be insufficient for reliable analysis")
    
    # Step 4: Perform parameter estimation
    print("\n4. Estimating precipitation parameters...")
    engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
    
    # Calculate monthly parameters
    monthly_params = engine.calculate_monthly_parameters()
    
    print("   Monthly Parameters:")
    print("   Month  P(W|W)  P(W|D)  Alpha   Beta")
    print("   " + "-" * 35)
    for month, params in monthly_params.items():
        print(f"   {month:2d}     {params.p_ww:.3f}   {params.p_wd:.3f}   {params.alpha:.2f}   {params.beta:.2f}")
    
    # Step 5: Generate parameter manifest
    print("\n5. Generating parameter manifest...")
    manifest = engine.generate_parameter_manifest()
    
    print(f"   Station ID: {manifest['metadata']['station_id']}")
    print(f"   Data period: {manifest['metadata']['data_period'][0]} to {manifest['metadata']['data_period'][1]}")
    print(f"   Wet day threshold: {manifest['metadata']['wet_day_threshold']} mm")
    
    # Step 6: Generate synthetic precipitation (stationary mode)
    print("\n6. Generating synthetic precipitation (stationary mode)...")
    sim = pg.SimulationEngine(manifest, trend_mode=False, random_seed=42)
    sim.initialize(datetime(2025, 1, 1), initial_wet_state=False)
    
    # Generate one year of synthetic data
    synthetic_data = []
    for day in range(365):
        precip = sim.step()
        synthetic_data.append(precip)
    
    synthetic_series = pd.Series(synthetic_data)
    
    print(f"   Generated {len(synthetic_data)} days of synthetic precipitation")
    print(f"   Synthetic mean: {synthetic_series.mean():.2f} mm")
    print(f"   Synthetic wet day frequency: {(synthetic_series > 0.001).mean():.1%}")
    print(f"   Annual total: {synthetic_series.sum():.1f} mm")
    
    # Step 7: Compare statistics
    print("\n7. Comparing historical vs synthetic statistics...")
    
    # Calculate monthly statistics
    historical_monthly = precip_data.groupby(precip_data.index.month).agg({
        'precipitation': ['mean', lambda x: (x > 0.001).mean()]
    }).round(3)
    
    # For synthetic data, we need to create dates
    synthetic_dates = pd.date_range(datetime(2025, 1, 1), periods=365, freq='D')
    synthetic_with_dates = pd.Series(synthetic_data, index=synthetic_dates)
    
    synthetic_monthly = synthetic_with_dates.groupby(synthetic_with_dates.index.month).agg([
        'mean', lambda x: (x > 0.001).mean()
    ]).round(3)
    
    print("   Monthly Comparison (Historical vs Synthetic):")
    print("   Month  Hist_Mean  Synth_Mean  Hist_WetFreq  Synth_WetFreq")
    print("   " + "-" * 55)
    
    for month in range(1, 13):
        hist_mean = precip_data[precip_data.index.month == month].mean()
        hist_wet_freq = (precip_data[precip_data.index.month == month] > 0.001).mean()
        
        if month <= len(synthetic_with_dates.groupby(synthetic_with_dates.index.month)):
            synth_month_data = synthetic_with_dates[synthetic_with_dates.index.month == month]
            synth_mean = synth_month_data.mean()
            synth_wet_freq = (synth_month_data > 0.001).mean()
        else:
            synth_mean = 0.0
            synth_wet_freq = 0.0
        
        print(f"   {month:2d}     {hist_mean:6.2f}     {synth_mean:6.2f}      {hist_wet_freq:6.3f}       {synth_wet_freq:6.3f}")
    
    # Step 8: Save results
    print("\n8. Saving results...")
    
    # Save parameter manifest
    import json
    with open('parameter_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print("   Parameter manifest saved to 'parameter_manifest.json'")
    
    # Save synthetic data
    synthetic_df = pd.DataFrame({
        'date': synthetic_dates,
        'precipitation': synthetic_data
    })
    synthetic_df.to_csv('synthetic_precipitation.csv', index=False)
    print("   Synthetic data saved to 'synthetic_precipitation.csv'")
    
    # Step 9: Create visualization (optional)
    try:
        print("\n9. Creating visualization...")
        create_comparison_plot(precip_data, synthetic_series)
        print("   Comparison plot saved to 'precipitation_comparison.png'")
    except ImportError:
        print("   Matplotlib not available, skipping visualization")
    
    print("\nAnalysis complete!")
    print("\nNext steps:")
    print("- Review parameter_manifest.json for detailed parameters")
    print("- Examine synthetic_precipitation.csv for generated data")
    print("- Try trend analysis with perform_sliding_window_analysis()")
    print("- Experiment with different random seeds for ensemble generation")

def create_comparison_plot(historical, synthetic):
    """Create comparison plots of historical vs synthetic data."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Historical vs Synthetic Precipitation Comparison', fontsize=14)
    
    # Daily time series (first 365 days)
    axes[0, 0].plot(historical.iloc[:365].values, alpha=0.7, label='Historical')
    axes[0, 0].plot(synthetic.values, alpha=0.7, label='Synthetic')
    axes[0, 0].set_title('Daily Time Series (First Year)')
    axes[0, 0].set_ylabel('Precipitation (mm)')
    axes[0, 0].legend()
    
    # Precipitation amount histograms (wet days only)
    hist_wet = historical[historical > 0.001]
    synth_wet = synthetic[synthetic > 0.001]
    
    axes[0, 1].hist(hist_wet, bins=30, alpha=0.7, label='Historical', density=True)
    axes[0, 1].hist(synth_wet, bins=30, alpha=0.7, label='Synthetic', density=True)
    axes[0, 1].set_title('Wet Day Amount Distribution')
    axes[0, 1].set_xlabel('Precipitation (mm)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # Monthly means
    hist_monthly = historical.groupby(historical.index.month).mean()
    synth_dates = pd.date_range(datetime(2025, 1, 1), periods=len(synthetic), freq='D')
    synth_with_dates = pd.Series(synthetic.values, index=synth_dates)
    synth_monthly = synth_with_dates.groupby(synth_with_dates.index.month).mean()
    
    months = range(1, 13)
    axes[1, 0].bar([m - 0.2 for m in months], hist_monthly, width=0.4, alpha=0.7, label='Historical')
    axes[1, 0].bar([m + 0.2 for m in months], synth_monthly, width=0.4, alpha=0.7, label='Synthetic')
    axes[1, 0].set_title('Monthly Mean Precipitation')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Mean Precipitation (mm)')
    axes[1, 0].legend()
    axes[1, 0].set_xticks(months)
    
    # Wet day frequency by month
    hist_wet_freq = (historical.groupby(historical.index.month).apply(lambda x: (x > 0.001).mean()))
    synth_wet_freq = (synth_with_dates.groupby(synth_with_dates.index.month).apply(lambda x: (x > 0.001).mean()))
    
    axes[1, 1].bar([m - 0.2 for m in months], hist_wet_freq, width=0.4, alpha=0.7, label='Historical')
    axes[1, 1].bar([m + 0.2 for m in months], synth_wet_freq, width=0.4, alpha=0.7, label='Synthetic')
    axes[1, 1].set_title('Monthly Wet Day Frequency')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Wet Day Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_xticks(months)
    
    plt.tight_layout()
    plt.savefig('precipitation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()