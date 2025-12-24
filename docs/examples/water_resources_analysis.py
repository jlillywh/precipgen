#!/usr/bin/env python3
"""
Water Resources Analysis for Precipitation Simulations

This script demonstrates how to use the enhanced precipitation simulations
for water resources applications including:
- Drought analysis
- Flood frequency analysis  
- Seasonal water availability
- Risk assessment for infrastructure planning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_drought_characteristics(precip_data, threshold_percentile=20):
    """
    Analyze drought characteristics from precipitation data.
    
    Args:
        precip_data: DataFrame with precipitation simulations
        threshold_percentile: Percentile below which conditions are considered drought
    
    Returns:
        Dictionary with drought statistics
    """
    # Calculate monthly totals
    monthly_totals = precip_data.resample('ME').sum()
    
    # Define drought threshold (e.g., 20th percentile of historical monthly totals)
    threshold = monthly_totals.quantile(threshold_percentile/100, axis=1)
    
    drought_stats = {}
    
    for col in monthly_totals.columns:
        monthly_series = monthly_totals[col]
        
        # Identify drought months
        drought_months = monthly_series < threshold
        
        # Calculate drought characteristics
        drought_events = []
        in_drought = False
        drought_start = None
        drought_duration = 0
        drought_severity = 0
        
        for date, is_drought in drought_months.items():
            if is_drought and not in_drought:
                # Start of drought
                in_drought = True
                drought_start = date
                drought_duration = 1
                drought_severity = threshold[date] - monthly_series[date]
            elif is_drought and in_drought:
                # Continuing drought
                drought_duration += 1
                drought_severity += threshold[date] - monthly_series[date]
            elif not is_drought and in_drought:
                # End of drought
                drought_events.append({
                    'start': drought_start,
                    'duration': drought_duration,
                    'severity': drought_severity
                })
                in_drought = False
        
        # Handle drought that continues to end of record
        if in_drought:
            drought_events.append({
                'start': drought_start,
                'duration': drought_duration,
                'severity': drought_severity
            })
        
        if drought_events:
            durations = [d['duration'] for d in drought_events]
            severities = [d['severity'] for d in drought_events]
            
            drought_stats[col] = {
                'n_events': len(drought_events),
                'mean_duration': np.mean(durations),
                'max_duration': np.max(durations),
                'mean_severity': np.mean(severities),
                'max_severity': np.max(severities)
            }
        else:
            drought_stats[col] = {
                'n_events': 0,
                'mean_duration': 0,
                'max_duration': 0,
                'mean_severity': 0,
                'max_severity': 0
            }
    
    return drought_stats

def analyze_extreme_events(precip_data, return_periods=[2, 5, 10, 25, 50, 100]):
    """
    Analyze extreme precipitation events and return periods.
    
    Args:
        precip_data: DataFrame with daily precipitation simulations
        return_periods: List of return periods to calculate (years)
    
    Returns:
        Dictionary with extreme event statistics
    """
    # Annual maximum daily precipitation
    annual_max = precip_data.resample('YE').max()
    
    extreme_stats = {}
    
    for col in annual_max.columns:
        annual_max_series = annual_max[col].dropna()
        
        if len(annual_max_series) > 0:
            # Sort in descending order
            sorted_max = np.sort(annual_max_series.values)[::-1]
            
            # Calculate return period estimates using plotting position
            n = len(sorted_max)
            plotting_positions = [(i + 1) / (n + 1) for i in range(n)]
            return_period_empirical = [1 / p for p in plotting_positions]
            
            # Interpolate for desired return periods
            return_levels = {}
            for rp in return_periods:
                if rp <= max(return_period_empirical):
                    return_levels[rp] = np.interp(rp, return_period_empirical[::-1], sorted_max[::-1])
                else:
                    # Extrapolate using simple linear trend (could use GEV distribution)
                    return_levels[rp] = np.nan
            
            extreme_stats[col] = {
                'annual_max_mean': annual_max_series.mean(),
                'annual_max_std': annual_max_series.std(),
                'return_levels': return_levels,
                'empirical_data': {
                    'values': sorted_max,
                    'return_periods': return_period_empirical
                }
            }
    
    return extreme_stats

def analyze_seasonal_water_availability(precip_data, seasons=None):
    """
    Analyze seasonal water availability patterns.
    
    Args:
        precip_data: DataFrame with daily precipitation simulations
        seasons: Dictionary defining seasons (default: standard 4 seasons)
    
    Returns:
        Dictionary with seasonal statistics
    """
    if seasons is None:
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5], 
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
    
    seasonal_stats = {}
    
    for season_name, months in seasons.items():
        # Filter data for season
        seasonal_data = precip_data[precip_data.index.month.isin(months)]
        
        # Calculate seasonal totals
        seasonal_totals = seasonal_data.resample('YE').sum()
        
        # Calculate statistics across realizations
        seasonal_stats[season_name] = {
            'mean_total': seasonal_totals.mean().mean(),
            'std_total': seasonal_totals.std().mean(),
            'cv': seasonal_totals.std().mean() / seasonal_totals.mean().mean(),
            'percentiles': {
                'p10': seasonal_totals.quantile(0.1, axis=1).mean(),
                'p25': seasonal_totals.quantile(0.25, axis=1).mean(),
                'p50': seasonal_totals.quantile(0.5, axis=1).mean(),
                'p75': seasonal_totals.quantile(0.75, axis=1).mean(),
                'p90': seasonal_totals.quantile(0.9, axis=1).mean()
            }
        }
    
    return seasonal_stats

def create_water_resources_report(simulation_results, output_dir='output'):
    """
    Create comprehensive water resources analysis report.
    
    Args:
        simulation_results: Dictionary of simulation results from different methods
        output_dir: Directory to save outputs
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("=== WATER RESOURCES ANALYSIS REPORT ===")
    
    for method, precip_data in simulation_results.items():
        print(f"\n--- {method.upper()} METHOD ---")
        
        # 1. Drought Analysis
        print("\n1. DROUGHT ANALYSIS")
        drought_stats = analyze_drought_characteristics(precip_data)
        
        # Summarize across realizations
        n_events = [stats['n_events'] for stats in drought_stats.values()]
        mean_durations = [stats['mean_duration'] for stats in drought_stats.values() if stats['n_events'] > 0]
        max_durations = [stats['max_duration'] for stats in drought_stats.values() if stats['n_events'] > 0]
        
        print(f"  Average drought events per 10-year period: {np.mean(n_events):.1f} ± {np.std(n_events):.1f}")
        if mean_durations:
            print(f"  Average drought duration: {np.mean(mean_durations):.1f} ± {np.std(mean_durations):.1f} months")
            print(f"  Maximum drought duration: {np.mean(max_durations):.1f} ± {np.std(max_durations):.1f} months")
        
        # 2. Extreme Events Analysis
        print("\n2. EXTREME EVENTS ANALYSIS")
        extreme_stats = analyze_extreme_events(precip_data)
        
        # Summarize return levels across realizations
        return_periods = [2, 5, 10, 25, 50]
        for rp in return_periods:
            return_levels = [stats['return_levels'].get(rp, np.nan) for stats in extreme_stats.values()]
            return_levels = [rl for rl in return_levels if not np.isnan(rl)]
            
            if return_levels:
                print(f"  {rp}-year return level: {np.mean(return_levels):.1f} ± {np.std(return_levels):.1f} mm/day")
        
        # 3. Seasonal Analysis
        print("\n3. SEASONAL WATER AVAILABILITY")
        seasonal_stats = analyze_seasonal_water_availability(precip_data)
        
        for season, stats in seasonal_stats.items():
            print(f"  {season}:")
            print(f"    Mean total: {stats['mean_total']:.0f} ± {stats['std_total']:.0f} mm")
            print(f"    Coefficient of variation: {stats['cv']:.2f}")
            print(f"    10th-90th percentile range: {stats['percentiles']['p10']:.0f} - {stats['percentiles']['p90']:.0f} mm")
        
        # Create plots
        create_water_resources_plots(precip_data, method, output_dir)

def create_water_resources_plots(precip_data, method_name, output_dir):
    """Create water resources specific plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Annual precipitation variability
    ax = axes[0, 0]
    annual_totals = precip_data.resample('YE').sum()
    
    # Box plot of annual totals across realizations
    annual_data = [annual_totals.iloc[i, :].values for i in range(len(annual_totals))]
    years = annual_totals.index.year
    
    bp = ax.boxplot(annual_data, positions=years, widths=0.6)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Precipitation (mm)')
    ax.set_title(f'{method_name.title()} - Annual Precipitation Variability')
    ax.grid(True, alpha=0.3)
    
    # 2. Seasonal distribution
    ax = axes[0, 1]
    seasons = {
        'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 
        'Summer': [6, 7, 8], 'Fall': [9, 10, 11]
    }
    
    seasonal_means = []
    seasonal_labels = []
    
    for season_name, months in seasons.items():
        seasonal_data = precip_data[precip_data.index.month.isin(months)]
        seasonal_totals = seasonal_data.resample('YE').sum()
        seasonal_means.append(seasonal_totals.values.flatten())
        seasonal_labels.append(season_name)
    
    ax.boxplot(seasonal_means, labels=seasonal_labels)
    ax.set_ylabel('Seasonal Precipitation (mm)')
    ax.set_title(f'{method_name.title()} - Seasonal Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. Wet spell analysis
    ax = axes[1, 0]
    
    # Calculate wet spell lengths for first realization
    daily_data = precip_data.iloc[:, 0]
    wet_days = daily_data > 0.1
    
    # Find wet spells
    wet_spells = []
    current_spell = 0
    
    for is_wet in wet_days:
        if is_wet:
            current_spell += 1
        else:
            if current_spell > 0:
                wet_spells.append(current_spell)
                current_spell = 0
    
    if current_spell > 0:
        wet_spells.append(current_spell)
    
    if wet_spells:
        ax.hist(wet_spells, bins=range(1, max(wet_spells) + 2), alpha=0.7, density=True)
        ax.set_xlabel('Wet Spell Length (days)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{method_name.title()} - Wet Spell Distribution')
        ax.grid(True, alpha=0.3)
    
    # 4. Monthly precipitation patterns
    ax = axes[1, 1]
    monthly_means = precip_data.groupby(precip_data.index.month).mean().mean(axis=1)
    monthly_stds = precip_data.groupby(precip_data.index.month).std().mean(axis=1)
    
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax.errorbar(months, monthly_means.values, yerr=monthly_stds.values, 
               marker='o', capsize=5, capthick=2)
    ax.set_xlabel('Month')
    ax.set_ylabel('Mean Daily Precipitation (mm)')
    ax.set_title(f'{method_name.title()} - Monthly Climatology')
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{method_name}_water_resources_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run water resources analysis."""
    
    # This assumes you've already run the enhanced_precip_simulation.py
    # and have results saved in the output directory
    
    output_dir = 'output'
    
    # Try to load existing simulation results
    try:
        # Load sample realizations (you would load your full results here)
        methods = ['wgen', 'bootstrap_doy', 'block_bootstrap']
        simulation_results = {}
        
        for method in methods:
            file_path = f'{output_dir}/{method}_sample_realizations.csv'
            if Path(file_path).exists():
                simulation_results[method] = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if simulation_results:
            print("Loaded existing simulation results.")
            create_water_resources_report(simulation_results, output_dir)
        else:
            print("No simulation results found. Please run enhanced_precip_simulation.py first.")
            
    except Exception as e:
        print(f"Error loading results: {e}")
        print("Please run enhanced_precip_simulation.py first to generate simulation data.")

if __name__ == "__main__":
    main()