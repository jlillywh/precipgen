#!/usr/bin/env python3
"""
Jupyter Notebook Template for Precipitation Simulation

Copy these cells into a Jupyter notebook for interactive precipitation modeling.
Each cell is designed to be run independently with clear outputs.
"""

# Cell 1: Setup and Imports
"""
import precipgen as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

print(f"PrecipGen version: {pg.__version__}")
"""

# Cell 2: Initialize and Load Data
"""
# Initialize simulator for your location
sim = pg.PrecipitationSimulator()

# Load historical data (replace with your coordinates)
latitude = 40.7608   # Salt Lake City
longitude = -111.8910

historical = sim.load_historical_data(latitude=latitude, longitude=longitude)

print(f"Station: {sim.station_info['name']}")
print(f"Distance: {sim.station_info['distance_km']:.1f} km")
print(f"Historical period: {historical.index.min().date()} to {historical.index.max().date()}")
print(f"Total days: {len(historical)}")

# Quick historical statistics
annual_hist = historical.resample('YE').sum()
print(f"\\nHistorical annual precipitation: {annual_hist.mean():.0f} ± {annual_hist.std():.0f} mm")
"""

# Cell 3: Calculate Parameters
"""
# Calculate monthly parameters
params = sim.calculate_monthly_parameters()

# Display parameters
print("Monthly Parameters:")
display(params.round(4))

# Plot seasonal patterns
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Wet day probability
axes[0].plot(params.index, params['p_w'], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Wet Day Probability')
axes[0].set_title('Seasonal Wet Day Probability')
axes[0].grid(True, alpha=0.3)

# Mean precipitation
axes[1].plot(params.index, params['mean_precip'], 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Mean Daily Precipitation (mm)')
axes[1].set_title('Seasonal Mean Precipitation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""

# Cell 4: Run Monte Carlo Simulation
"""
# Run simulation (adjust parameters as needed)
results = sim.run_monte_carlo_simulation(
    start_date='2026-01-01',
    end_date='2035-12-31',     # 10 years
    n_realizations=100,        # 100 realizations
    methods=['wgen', 'bootstrap_doy', 'block_bootstrap']
)

print("Simulation complete!")
print(f"Generated {len(results)} methods with {results['wgen'].shape[1]} realizations each")
print(f"Simulation period: {results['wgen'].index.min().date()} to {results['wgen'].index.max().date()}")
"""

# Cell 5: Quick Results Comparison
"""
# Compare methods
print("=== METHOD COMPARISON ===")

historical_annual = historical.resample('YE').sum()
print(f"Historical annual mean: {historical_annual.mean():.0f} mm")

for method, sims in results.items():
    annual_sims = sims.resample('YE').sum()
    print(f"{method.upper()}:")
    print(f"  Mean: {annual_sims.mean().mean():.0f} mm")
    print(f"  Std:  {annual_sims.std().mean():.0f} mm")
    print(f"  Range: {annual_sims.min().min():.0f} - {annual_sims.max().max():.0f} mm")
"""

# Cell 6: Visualization
"""
# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Annual totals comparison
ax = axes[0, 0]
annual_data = []
labels = ['Historical']
annual_data.append(historical_annual.values)

for method in results.keys():
    annual_sims = results[method].resample('YE').sum()
    annual_data.append(annual_sims.values.flatten())
    labels.append(method.title())

ax.boxplot(annual_data, labels=labels)
ax.set_ylabel('Annual Precipitation (mm)')
ax.set_title('Annual Precipitation Distribution')
ax.grid(True, alpha=0.3)

# Monthly climatology
ax = axes[0, 1]
hist_monthly = historical.groupby(historical.index.month).mean()
months = range(1, 13)

ax.plot(months, hist_monthly.values, 'ko-', label='Historical', linewidth=2)

for method in results.keys():
    monthly_clim = results[method].groupby(results[method].index.month).mean().mean(axis=1)
    ax.plot(months, monthly_clim.values, 'o-', label=method.title(), alpha=0.7)

ax.set_xlabel('Month')
ax.set_ylabel('Mean Daily Precipitation (mm)')
ax.set_title('Monthly Climatology')
ax.legend()
ax.grid(True, alpha=0.3)

# Time series (first method, first 5 realizations)
ax = axes[1, 0]
method_name = list(results.keys())[0]
annual_sims = results[method_name].resample('YE').sum()
years = annual_sims.index.year

for i in range(min(5, annual_sims.shape[1])):
    ax.plot(years, annual_sims.iloc[:, i], alpha=0.6, linewidth=1)

ax.plot(historical_annual.index.year, historical_annual.values, 'ko-', 
        label='Historical', alpha=0.8, linewidth=2)
ax.set_xlabel('Year')
ax.set_ylabel('Annual Precipitation (mm)')
ax.set_title(f'{method_name.title()} - Sample Realizations')
ax.legend()
ax.grid(True, alpha=0.3)

# Wet day frequency
ax = axes[1, 1]
hist_wet_freq = (historical > 0.1).groupby(historical.index.month).mean()
ax.plot(months, hist_wet_freq.values, 'ko-', label='Historical', linewidth=2)

for method in results.keys():
    wet_freq = (results[method] > 0.1).groupby(results[method].index.month).mean().mean(axis=1)
    ax.plot(months, wet_freq.values, 'o-', label=method.title(), alpha=0.7)

ax.set_xlabel('Month')
ax.set_ylabel('Wet Day Frequency')
ax.set_title('Seasonal Wet Day Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""

# Cell 7: Water Resources Analysis
"""
# Get specialized water resources analysis
water_analysis = sim.get_water_resources_analysis()

print("=== WATER RESOURCES ANALYSIS ===")

for method in results.keys():
    print(f"\\n--- {method.upper()} METHOD ---")
    
    # Drought analysis
    drought_stats = water_analysis[method]['drought']['summary']
    print(f"Drought events per 10-year period: {drought_stats['events_per_period']['mean']:.1f} ± {drought_stats['events_per_period']['std']:.1f}")
    
    if 'duration_months' in drought_stats:
        print(f"Average drought duration: {drought_stats['duration_months']['mean']:.1f} months")
        print(f"Maximum drought duration: {drought_stats['duration_months']['max_observed']:.0f} months")
    
    # Extreme events
    extreme_stats = water_analysis[method]['extreme_events']['summary']
    for rp in [2, 5, 10, 25]:
        key = f'{rp}_year_return_level'
        if key in extreme_stats:
            stats = extreme_stats[key]
            print(f"{rp}-year return level: {stats['mean']:.1f} ± {stats['std']:.1f} mm/day")
    
    # Seasonal analysis
    seasonal_stats = water_analysis[method]['seasonal']
    print("Seasonal totals (mm):")
    for season, stats in seasonal_stats.items():
        print(f"  {season}: {stats['mean_total']:.0f} ± {stats['std_total']:.0f}")
"""

# Cell 8: Export Results
"""
# Export results for further analysis
sim.export_results(output_dir='simulation_output')

# Also save some custom analysis
output_dir = 'simulation_output'

# Save annual statistics summary
annual_summary = []
for method in results.keys():
    annual_sims = results[method].resample('YE').sum()
    for year in annual_sims.index.year:
        year_data = annual_sims.loc[f'{year}']
        annual_summary.append({
            'method': method,
            'year': year,
            'mean': year_data.mean(),
            'std': year_data.std(),
            'min': year_data.min(),
            'max': year_data.max(),
            'p10': year_data.quantile(0.1),
            'p90': year_data.quantile(0.9)
        })

annual_df = pd.DataFrame(annual_summary)
annual_df.to_csv(f'{output_dir}/annual_summary_all_methods.csv', index=False)

print(f"Results exported to '{output_dir}/' directory")
print("Files created:")
print("- monthly_parameters.csv")
print("- *_sample_realizations.csv (for each method)")
print("- *_annual_statistics.csv (for each method)")
print("- annual_summary_all_methods.csv")
print("- precipitation_analysis.png")
"""

# Cell 9: Custom Analysis (Optional)
"""
# Example: Calculate return periods for annual maxima
method = 'wgen'  # Choose method
annual_max = results[method].resample('YE').max()

# Calculate empirical return periods
return_periods = []
for col in annual_max.columns:
    series = annual_max[col].dropna().sort_values(ascending=False)
    n = len(series)
    rp = [(i + 1) / (n + 1) for i in range(n)]
    return_periods.append(pd.DataFrame({
        'return_period': [1/p for p in rp],
        'value': series.values,
        'realization': col
    }))

rp_df = pd.concat(return_periods, ignore_index=True)

# Plot return period curves
plt.figure(figsize=(10, 6))
for col in annual_max.columns[:10]:  # Plot first 10 realizations
    subset = rp_df[rp_df['realization'] == col]
    plt.semilogx(subset['return_period'], subset['value'], alpha=0.3, color='blue')

# Add mean curve
mean_curve = rp_df.groupby('return_period')['value'].mean().reset_index()
plt.semilogx(mean_curve['return_period'], mean_curve['value'], 'r-', linewidth=3, label='Mean')

plt.xlabel('Return Period (years)')
plt.ylabel('Annual Maximum Daily Precipitation (mm)')
plt.title(f'{method.title()} - Return Period Analysis')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("Custom analysis complete!")
"""

def create_notebook_template():
    """Create a .ipynb file with the above cells."""
    
    cells = [
        "# Setup and Imports",
        "# Initialize and Load Data", 
        "# Calculate Parameters",
        "# Run Monte Carlo Simulation",
        "# Quick Results Comparison",
        "# Visualization",
        "# Water Resources Analysis", 
        "# Export Results",
        "# Custom Analysis (Optional)"
    ]
    
    print("Jupyter Notebook Template")
    print("=" * 50)
    print("Copy the code blocks above into separate Jupyter notebook cells.")
    print("Each cell is designed to run independently with clear outputs.")
    print("\nRecommended cell structure:")
    for i, cell in enumerate(cells, 1):
        print(f"Cell {i}: {cell}")

if __name__ == "__main__":
    create_notebook_template()