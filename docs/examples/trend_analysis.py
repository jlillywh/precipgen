#!/usr/bin/env python3
"""
Trend Analysis Example

This example demonstrates non-stationary precipitation analysis and simulation:
1. Generate data with embedded trends
2. Perform sliding window analysis
3. Detect and quantify trends
4. Generate non-stationary synthetic precipitation

Author: PrecipGen Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import precipgen as pg

def generate_trending_data(years=50, seed=42):
    """
    Generate precipitation data with embedded trends.
    
    This creates data where precipitation parameters change over time,
    simulating climate change effects.
    """
    np.random.seed(seed)
    
    # Create date range
    start_date = datetime(1970, 1, 1)
    end_date = datetime(1970 + years, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    precipitation = []
    is_wet_yesterday = False
    
    for i, date in enumerate(dates):
        month = date.month
        year_fraction = i / len(dates)  # 0 to 1 over the time period
        
        # Base seasonal parameters
        if month in [12, 1, 2]:  # Winter
            base_p_ww, base_p_wd = 0.6, 0.3
            base_alpha, base_beta = 1.2, 8.0
            # Winter trends: increasing wet persistence, decreasing amounts
            trend_p_ww = 0.1 * year_fraction  # +0.1 over full period
            trend_p_wd = 0.05 * year_fraction
            trend_alpha = -0.2 * year_fraction
            trend_beta = -2.0 * year_fraction
        elif month in [6, 7, 8]:  # Summer
            base_p_ww, base_p_wd = 0.4, 0.15
            base_alpha, base_beta = 1.5, 5.0
            # Summer trends: decreasing wet persistence, increasing intensity
            trend_p_ww = -0.05 * year_fraction
            trend_p_wd = -0.02 * year_fraction
            trend_alpha = 0.3 * year_fraction
            trend_beta = 1.0 * year_fraction
        else:  # Spring/Fall
            base_p_ww, base_p_wd = 0.5, 0.25
            base_alpha, base_beta = 1.3, 6.5
            # Moderate trends
            trend_p_ww = 0.02 * year_fraction
            trend_p_wd = 0.01 * year_fraction
            trend_alpha = 0.1 * year_fraction
            trend_beta = 0.5 * year_fraction
        
        # Apply trends with bounds
        p_ww = max(0.1, min(0.9, base_p_ww + trend_p_ww))
        p_wd = max(0.05, min(0.8, base_p_wd + trend_p_wd))
        alpha = max(0.5, base_alpha + trend_alpha)
        beta = max(1.0, base_beta + trend_beta)
        
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
    """Main trend analysis workflow demonstration."""
    
    print("PrecipGen Trend Analysis Example")
    print("=" * 40)
    
    # Step 1: Generate trending data
    print("\n1. Generating precipitation data with embedded trends...")
    precip_data = generate_trending_data(years=50, seed=42)
    
    print(f"   Data period: {precip_data.index[0].date()} to {precip_data.index[-1].date()}")
    print(f"   Total days: {len(precip_data)}")
    print(f"   Overall mean: {precip_data.mean():.2f} mm")
    
    # Compare early vs late periods
    early_period = precip_data[precip_data.index.year <= 1980]
    late_period = precip_data[precip_data.index.year >= 2010]
    
    print(f"   Early period (1970-1980) mean: {early_period.mean():.2f} mm")
    print(f"   Late period (2010-2020) mean: {late_period.mean():.2f} mm")
    print(f"   Change: {late_period.mean() - early_period.mean():.2f} mm")
    
    # Step 2: Configure for trend analysis
    print("\n2. Configuring for trend analysis...")
    config = pg.PrecipGenConfig({
        'analysis': {
            'wet_day_threshold': 0.001,
            'sliding_window_years': 20,  # 20-year windows
            'trend_significance_level': 0.05
        }
    })
    
    # Step 3: Perform sliding window analysis
    print("\n3. Performing sliding window analysis...")
    engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
    
    # This may take a moment for 50 years of data
    print("   Computing parameters for overlapping 20-year windows...")
    window_analysis = engine.perform_sliding_window_analysis(window_years=20)
    
    print(f"   Analyzed {len(window_analysis.results)} overlapping windows")
    print(f"   Window centers: {window_analysis.results[0].center_year} to {window_analysis.results[-1].center_year}")
    
    # Step 4: Extract and analyze trends
    print("\n4. Extracting trends from sliding window results...")
    trend_analysis = engine.extract_trends(window_analysis)
    
    # Display significant trends
    print("   Significant trends detected (p < 0.05):")
    print("   Parameter    Season    Slope/year    P-value    R²")
    print("   " + "-" * 50)
    
    significant_trends = []
    for param_name in ['p_ww', 'p_wd', 'alpha', 'beta']:
        if param_name in trend_analysis.seasonal_slopes:
            for season in trend_analysis.seasonal_slopes[param_name]:
                slope = trend_analysis.seasonal_slopes[param_name][season]
                p_value = trend_analysis.significance_tests[param_name][season]
                r_squared = trend_analysis.r_squared[param_name][season] if hasattr(trend_analysis, 'r_squared') else 0.0
                
                if p_value < 0.05:
                    print(f"   {param_name:10s}   {season:8s}   {slope:8.5f}     {p_value:.3f}    {r_squared:.3f}")
                    significant_trends.append((param_name, season, slope, p_value))
    
    if not significant_trends:
        print("   No statistically significant trends detected")
    
    # Step 5: Generate complete parameter manifest with trends
    print("\n5. Generating parameter manifest with trend information...")
    manifest = engine.generate_parameter_manifest()
    
    print(f"   Manifest includes trend analysis: {'trend_analysis' in manifest}")
    if 'trend_analysis' in manifest:
        print(f"   Seasonal trend slopes available: {len(manifest['trend_analysis']['seasonal_slopes'])} seasons")
    
    # Step 6: Stationary simulation (baseline)
    print("\n6. Generating stationary synthetic precipitation...")
    sim_stationary = pg.SimulationEngine(manifest, trend_mode=False, random_seed=42)
    sim_stationary.initialize(datetime(2025, 1, 1))
    
    # Generate 10 years of stationary data
    stationary_data = []
    for year in range(10):
        year_data = []
        for day in range(365):
            precip = sim_stationary.step()
            year_data.append(precip)
        annual_total = sum(year_data)
        stationary_data.append(annual_total)
        if year < 3:  # Show first few years
            print(f"   Stationary Year {year+1}: {annual_total:.1f} mm")
    
    stationary_mean = np.mean(stationary_data)
    print(f"   Stationary 10-year mean: {stationary_mean:.1f} mm/year")
    
    # Step 7: Non-stationary simulation with trends
    print("\n7. Generating non-stationary synthetic precipitation...")
    sim_trending = pg.SimulationEngine(manifest, trend_mode=True, random_seed=42)
    sim_trending.initialize(datetime(2025, 1, 1))
    
    # Generate 50 years with parameter drift
    trending_data = []
    for year in range(50):
        year_data = []
        for day in range(365):
            precip = sim_trending.step()
            year_data.append(precip)
        annual_total = sum(year_data)
        trending_data.append(annual_total)
        if year < 3 or year >= 47:  # Show first and last few years
            print(f"   Trending Year {year+1}: {annual_total:.1f} mm")
    
    # Analyze trend in synthetic data
    years = np.arange(1, 51)
    trend_slope = np.polyfit(years, trending_data, 1)[0]
    print(f"   Synthetic trend: {trend_slope:.2f} mm/year over 50 years")
    print(f"   Total change: {trend_slope * 50:.1f} mm/year")
    
    # Step 8: Compare simulation modes
    print("\n8. Comparing stationary vs non-stationary simulation...")
    
    # Early vs late periods in trending simulation
    early_trending = np.mean(trending_data[:10])
    late_trending = np.mean(trending_data[-10:])
    
    print(f"   Stationary simulation mean: {stationary_mean:.1f} mm/year")
    print(f"   Non-stationary early period (years 1-10): {early_trending:.1f} mm/year")
    print(f"   Non-stationary late period (years 41-50): {late_trending:.1f} mm/year")
    print(f"   Non-stationary change: {late_trending - early_trending:.1f} mm/year")
    
    # Step 9: Parameter evolution analysis
    print("\n9. Analyzing parameter evolution during simulation...")
    
    # Reset simulation and track parameter changes
    sim_tracking = pg.SimulationEngine(manifest, trend_mode=True, random_seed=42)
    sim_tracking.initialize(datetime(2025, 1, 1))
    
    # Sample parameters at different time points
    parameter_evolution = []
    sample_years = [0, 10, 25, 40, 49]
    
    for target_year in sample_years:
        # Advance simulation to target year
        for day in range(target_year * 365):
            sim_tracking.step()
        
        # Get current state and parameters
        current_state = sim_tracking.get_current_state()
        current_params = current_state.current_parameters
        
        parameter_evolution.append({
            'year': target_year + 1,
            'p_ww': current_params.p_ww,
            'p_wd': current_params.p_wd,
            'alpha': current_params.alpha,
            'beta': current_params.beta
        })
        
        # Reset for next sample
        sim_tracking.reset(datetime(2025, 1, 1))
    
    print("   Parameter evolution over simulation period:")
    print("   Year   P(W|W)   P(W|D)   Alpha    Beta")
    print("   " + "-" * 38)
    for params in parameter_evolution:
        print(f"   {params['year']:2d}     {params['p_ww']:.3f}    {params['p_wd']:.3f}    {params['alpha']:.2f}    {params['beta']:.2f}")
    
    # Step 10: Save results and create visualizations
    print("\n10. Saving results...")
    
    # Save trend analysis results
    trend_results = {
        'window_analysis': {
            'window_years': 20,
            'num_windows': len(window_analysis.results),
            'center_years': [r.center_year for r in window_analysis.results]
        },
        'significant_trends': [
            {
                'parameter': param,
                'season': season,
                'slope_per_year': slope,
                'p_value': p_val
            }
            for param, season, slope, p_val in significant_trends
        ],
        'simulation_comparison': {
            'stationary_mean': float(stationary_mean),
            'nonstationary_early': float(early_trending),
            'nonstationary_late': float(late_trending),
            'synthetic_trend_slope': float(trend_slope)
        }
    }
    
    import json
    with open('trend_analysis_results.json', 'w') as f:
        json.dump(trend_results, f, indent=2)
    print("   Trend analysis results saved to 'trend_analysis_results.json'")
    
    # Save synthetic data
    trending_df = pd.DataFrame({
        'year': range(1, 51),
        'annual_precipitation': trending_data
    })
    trending_df.to_csv('nonstationary_precipitation.csv', index=False)
    print("   Non-stationary synthetic data saved to 'nonstationary_precipitation.csv'")
    
    # Create visualization
    try:
        print("   Creating trend visualization...")
        create_trend_plots(precip_data, window_analysis, trending_data, stationary_data)
        print("   Trend plots saved to 'trend_analysis.png'")
    except ImportError:
        print("   Matplotlib not available, skipping visualization")
    
    print("\nTrend analysis complete!")
    print("\nKey findings:")
    if significant_trends:
        print(f"- Detected {len(significant_trends)} statistically significant trends")
        print(f"- Synthetic simulation shows {trend_slope:.2f} mm/year trend")
    else:
        print("- No significant trends detected in this synthetic dataset")
    print("- Non-stationary simulation successfully incorporates parameter drift")
    print("- Parameter evolution tracked over 50-year simulation period")

def create_trend_plots(historical_data, window_analysis, trending_data, stationary_data):
    """Create comprehensive trend analysis plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Precipitation Trend Analysis', fontsize=16)
    
    # Plot 1: Historical data with trend
    years = historical_data.index.year
    annual_totals = historical_data.groupby(years).sum()
    
    axes[0, 0].plot(annual_totals.index, annual_totals.values, 'b-', alpha=0.7, label='Annual Total')
    
    # Add trend line
    x = annual_totals.index - annual_totals.index[0]
    trend_coef = np.polyfit(x, annual_totals.values, 1)
    trend_line = np.poly1d(trend_coef)
    axes[0, 0].plot(annual_totals.index, trend_line(x), 'r--', linewidth=2, 
                   label=f'Trend: {trend_coef[0]:.1f} mm/year')
    
    axes[0, 0].set_title('Historical Annual Precipitation')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Annual Precipitation (mm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter evolution from sliding windows
    if window_analysis.results:
        center_years = [r.center_year for r in window_analysis.results]
        
        # Extract P(W|W) for January as example
        jan_pww = []
        for result in window_analysis.results:
            if 1 in result.parameters:
                jan_pww.append(result.parameters[1].p_ww)
            else:
                jan_pww.append(np.nan)
        
        axes[0, 1].plot(center_years, jan_pww, 'go-', alpha=0.7, label='January P(W|W)')
        
        # Add trend line if enough data
        if len([x for x in jan_pww if not np.isnan(x)]) > 5:
            valid_indices = ~np.isnan(jan_pww)
            if np.any(valid_indices):
                trend_coef = np.polyfit(np.array(center_years)[valid_indices], 
                                      np.array(jan_pww)[valid_indices], 1)
                trend_line = np.poly1d(trend_coef)
                axes[0, 1].plot(center_years, trend_line(np.array(center_years) - center_years[0]), 
                               'r--', linewidth=2, label=f'Trend: {trend_coef[0]:.5f}/year')
        
        axes[0, 1].set_title('Parameter Evolution (Sliding Windows)')
        axes[0, 1].set_xlabel('Window Center Year')
        axes[0, 1].set_ylabel('P(W|W) January')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Stationary vs Non-stationary simulation
    sim_years = range(1, len(trending_data) + 1)
    
    axes[1, 0].plot(sim_years, trending_data, 'b-', alpha=0.7, label='Non-stationary')
    
    # Add stationary baseline
    stationary_mean = np.mean(stationary_data)
    axes[1, 0].axhline(y=stationary_mean, color='r', linestyle='--', linewidth=2, 
                      label=f'Stationary mean: {stationary_mean:.0f} mm')
    
    # Add trend line to non-stationary
    trend_coef = np.polyfit(sim_years, trending_data, 1)
    trend_line = np.poly1d(trend_coef)
    axes[1, 0].plot(sim_years, trend_line(np.array(sim_years) - 1), 'g--', linewidth=2,
                   label=f'Trend: {trend_coef[0]:.1f} mm/year')
    
    axes[1, 0].set_title('Synthetic Precipitation Comparison')
    axes[1, 0].set_xlabel('Simulation Year')
    axes[1, 0].set_ylabel('Annual Precipitation (mm)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Trend magnitude comparison
    periods = ['Early\n(1970-1980)', 'Late\n(2010-2020)', 'Synth Early\n(Years 1-10)', 'Synth Late\n(Years 41-50)']
    
    early_hist = historical_data[historical_data.index.year <= 1980].groupby(historical_data[historical_data.index.year <= 1980].index.year).sum().mean()
    late_hist = historical_data[historical_data.index.year >= 2010].groupby(historical_data[historical_data.index.year >= 2010].index.year).sum().mean()
    early_synth = np.mean(trending_data[:10])
    late_synth = np.mean(trending_data[-10:])
    
    values = [early_hist, late_hist, early_synth, late_synth]
    colors = ['lightblue', 'darkblue', 'lightgreen', 'darkgreen']
    
    bars = axes[1, 1].bar(periods, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Period Comparison')
    axes[1, 1].set_ylabel('Mean Annual Precipitation (mm)')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{value:.0f}', ha='center', va='bottom')
    
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()