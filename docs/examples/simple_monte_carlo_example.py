#!/usr/bin/env python3
"""
Simple Monte Carlo Precipitation Simulation Example

This example shows how to use the integrated PrecipitationSimulator
class for quick and easy Monte Carlo precipitation modeling.

Perfect for Jupyter notebooks!
"""

import precipgen as pg
import pandas as pd
import numpy as np

def main():
    """Simple example of Monte Carlo precipitation simulation."""
    
    print("=== SIMPLE MONTE CARLO PRECIPITATION SIMULATION ===")
    
    # 1. Initialize simulator (one line!)
    sim = pg.PrecipitationSimulator()
    
    # 2. Load data for Salt Lake City area (one line!)
    historical = sim.load_historical_data(latitude=40.7608, longitude=-111.8910)
    print(f"Loaded {len(historical)} days of historical data")
    
    # 3. Calculate parameters (one line!)
    params = sim.calculate_monthly_parameters()
    print("\nMonthly Parameters:")
    print(params[['p_w', 'alpha', 'beta']].round(3))
    
    # 4. Run Monte Carlo simulation (one line!)
    results = sim.run_monte_carlo_simulation(
        start_date='2026-01-01',
        end_date='2030-12-31',  # 5 years for quick demo
        n_realizations=50,      # 50 realizations for speed
        methods=['wgen', 'bootstrap_doy']  # Two methods
    )
    
    # 5. Quick analysis (one line!)
    analysis = sim.analyze_results(save_plots=True)
    
    # 6. Get water resources analysis (one line!)
    water_analysis = sim.get_water_resources_analysis()
    
    # Print some key results
    print("\n=== QUICK RESULTS SUMMARY ===")
    
    for method in results.keys():
        annual_totals = results[method].resample('YE').sum()
        print(f"\n{method.upper()} Method:")
        print(f"  Mean annual precipitation: {annual_totals.mean().mean():.0f} ± {annual_totals.std().mean():.0f} mm")
        print(f"  Range: {annual_totals.min().min():.0f} - {annual_totals.max().max():.0f} mm")
        
        # Drought info
        drought_info = water_analysis[method]['drought']['summary']
        print(f"  Average drought events per 5-year period: {drought_info['events_per_period']['mean']:.1f}")
    
    # 7. Export results (one line!)
    sim.export_results(output_dir='output')
    
    print(f"\nResults exported to 'output/' directory")
    print("Analysis complete!")
    
    return sim, results, analysis

# For Jupyter notebook usage:
def quick_simulation(latitude, longitude, years=10, n_realizations=100):
    """
    Ultra-simple function for Jupyter notebooks.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude  
        years: Number of years to simulate
        n_realizations: Number of Monte Carlo realizations
        
    Returns:
        Tuple of (simulator, results, analysis)
    """
    sim = pg.PrecipitationSimulator()
    sim.load_historical_data(latitude=latitude, longitude=longitude)
    sim.calculate_monthly_parameters()
    
    end_year = 2025 + years
    results = sim.run_monte_carlo_simulation(
        start_date='2026-01-01',
        end_date=f'{end_year}-12-31',
        n_realizations=n_realizations
    )
    
    analysis = sim.analyze_results()
    return sim, results, analysis

if __name__ == "__main__":
    simulator, simulation_results, analysis_results = main()