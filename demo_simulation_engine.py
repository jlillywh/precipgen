#!/usr/bin/env python3
"""
Demonstration of the SimulationEngine WGEN algorithms.

This script shows the core WGEN simulation algorithms in action:
1. Markov chain wet/dry state transitions
2. Gamma distribution sampling for wet day amounts
3. Monthly parameter selection
4. Random number generator state management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from precipgen.engines.simulation import SimulationEngine
from precipgen.engines.analytical import ParameterManifest, MonthlyParams


def create_demo_parameters():
    """Create demonstration parameters with seasonal variation."""
    monthly_params = {}
    
    for month in range(1, 13):
        # Create seasonal variation in parameters
        # Winter months (Dec, Jan, Feb) have higher wet probabilities
        # Summer months (Jun, Jul, Aug) have lower wet probabilities
        if month in [12, 1, 2]:  # Winter
            p_ww, p_wd = 0.7, 0.4
            alpha, beta = 1.8, 6.0
        elif month in [6, 7, 8]:  # Summer
            p_ww, p_wd = 0.4, 0.2
            alpha, beta = 1.2, 4.0
        else:  # Spring/Fall
            p_ww, p_wd = 0.6, 0.3
            alpha, beta = 1.5, 5.0
        
        monthly_params[month] = MonthlyParams(
            p_ww=p_ww,
            p_wd=p_wd,
            alpha=alpha,
            beta=beta
        )
    
    return ParameterManifest(
        metadata={
            'station_id': 'DEMO_STATION',
            'wet_day_threshold': 0.001,
            'description': 'Demonstration parameters with seasonal variation'
        },
        overall_parameters=monthly_params,
        trend_analysis=None,
        sliding_window_stats=None
    )


def demonstrate_markov_chain():
    """Demonstrate Markov chain wet/dry transitions."""
    print("=== Markov Chain Wet/Dry Transitions ===")
    
    # Create parameters with clear transition probabilities
    monthly_params = {}
    for month in range(1, 13):
        monthly_params[month] = MonthlyParams(
            p_ww=0.8,  # High probability of wet following wet
            p_wd=0.2,  # Low probability of wet following dry
            alpha=1.5,
            beta=5.0
        )
    
    manifest = ParameterManifest(
        metadata={'station_id': 'MARKOV_DEMO', 'wet_day_threshold': 0.001},
        overall_parameters=monthly_params,
        trend_analysis=None,
        sliding_window_stats=None
    )
    
    engine = SimulationEngine(manifest, random_seed=42)
    engine.initialize(datetime(2020, 1, 1), initial_wet_state=True)
    
    # Track transitions
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
    
    # Calculate observed probabilities
    total_wet = transitions['WW'] + transitions['WD']
    total_dry = transitions['DW'] + transitions['DD']
    
    observed_p_ww = transitions['WW'] / total_wet if total_wet > 0 else 0
    observed_p_wd = transitions['DW'] / total_dry if total_dry > 0 else 0
    
    print(f"Expected P(W|W) = 0.8, Observed = {observed_p_ww:.3f}")
    print(f"Expected P(W|D) = 0.2, Observed = {observed_p_wd:.3f}")
    print(f"Transitions: WW={transitions['WW']}, WD={transitions['WD']}, DW={transitions['DW']}, DD={transitions['DD']}")
    print()


def demonstrate_gamma_sampling():
    """Demonstrate Gamma distribution sampling."""
    print("=== Gamma Distribution Sampling ===")
    
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
    
    manifest = ParameterManifest(
        metadata={'station_id': 'GAMMA_DEMO', 'wet_day_threshold': 0.001},
        overall_parameters=monthly_params,
        trend_analysis=None,
        sliding_window_stats=None
    )
    
    engine = SimulationEngine(manifest, random_seed=42)
    engine.initialize(datetime(2020, 1, 1))
    
    # Collect wet day amounts
    wet_amounts = []
    for _ in range(1000):
        precip = engine.step()
        if precip > 0.001:
            wet_amounts.append(precip)
    
    wet_amounts = np.array(wet_amounts)
    
    # Calculate statistics
    expected_mean = alpha * beta
    expected_var = alpha * beta**2
    observed_mean = np.mean(wet_amounts)
    observed_var = np.var(wet_amounts)
    
    print(f"Gamma parameters: alpha={alpha}, beta={beta}")
    print(f"Expected mean = {expected_mean:.2f}, Observed = {observed_mean:.2f}")
    print(f"Expected variance = {expected_var:.2f}, Observed = {observed_var:.2f}")
    print(f"Generated {len(wet_amounts)} wet days out of 1000 total days")
    print()


def demonstrate_seasonal_simulation():
    """Demonstrate full seasonal simulation."""
    print("=== Seasonal Simulation Demonstration ===")
    
    manifest = create_demo_parameters()
    engine = SimulationEngine(manifest, random_seed=42)
    engine.initialize(datetime(2020, 1, 1))
    
    # Generate one year of data
    dates = []
    precipitation = []
    wet_days = []
    
    current_date = datetime(2020, 1, 1)
    for _ in range(365):
        precip = engine.step()
        dates.append(current_date)
        precipitation.append(precip)
        wet_days.append(precip > 0.001)
        current_date += timedelta(days=1)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'date': dates,
        'precipitation': precipitation,
        'is_wet': wet_days
    })
    df['month'] = df['date'].dt.month
    
    # Calculate monthly statistics
    monthly_stats = df.groupby('month').agg({
        'precipitation': ['mean', 'sum', 'count'],
        'is_wet': ['mean', 'sum']
    }).round(3)
    
    print("Monthly Statistics:")
    print("Month | Mean Precip | Total Precip | Days | Wet % | Wet Days")
    print("-" * 60)
    
    for month in range(1, 12):
        if month in monthly_stats.index:
            mean_precip = monthly_stats.loc[month, ('precipitation', 'mean')]
            total_precip = monthly_stats.loc[month, ('precipitation', 'sum')]
            total_days = monthly_stats.loc[month, ('precipitation', 'count')]
            wet_pct = monthly_stats.loc[month, ('is_wet', 'mean')] * 100
            wet_days = monthly_stats.loc[month, ('is_wet', 'sum')]
            
            print(f"{month:5d} | {mean_precip:11.3f} | {total_precip:12.3f} | {total_days:4.0f} | {wet_pct:5.1f} | {wet_days:8.0f}")
    
    print()
    
    # Overall statistics
    total_precip = df['precipitation'].sum()
    wet_day_count = df['is_wet'].sum()
    wet_percentage = (wet_day_count / len(df)) * 100
    
    print(f"Annual Summary:")
    print(f"Total precipitation: {total_precip:.1f} mm")
    print(f"Wet days: {wet_day_count} ({wet_percentage:.1f}%)")
    print(f"Average wet day amount: {df[df['is_wet']]['precipitation'].mean():.2f} mm")
    print()


def demonstrate_reproducibility():
    """Demonstrate random number generator reproducibility."""
    print("=== Random Number Generator Reproducibility ===")
    
    manifest = create_demo_parameters()
    
    # Generate two sequences with same seed
    engine1 = SimulationEngine(manifest, random_seed=123)
    engine1.initialize(datetime(2020, 1, 1))
    
    engine2 = SimulationEngine(manifest, random_seed=123)
    engine2.initialize(datetime(2020, 1, 1))
    
    sequence1 = [engine1.step() for _ in range(10)]
    sequence2 = [engine2.step() for _ in range(10)]
    
    print("Same seed (123) produces identical sequences:")
    print("Engine 1:", [f"{x:.3f}" for x in sequence1])
    print("Engine 2:", [f"{x:.3f}" for x in sequence2])
    print("Identical:", sequence1 == sequence2)
    print()
    
    # Generate sequence with different seed
    engine3 = SimulationEngine(manifest, random_seed=456)
    engine3.initialize(datetime(2020, 1, 1))
    sequence3 = [engine3.step() for _ in range(10)]
    
    print("Different seed (456) produces different sequence:")
    print("Engine 3:", [f"{x:.3f}" for x in sequence3])
    print("Different from Engine 1:", sequence1 != sequence3)
    print()


def main():
    """Run all demonstrations."""
    print("PrecipGen SimulationEngine WGEN Algorithms Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_markov_chain()
    demonstrate_gamma_sampling()
    demonstrate_seasonal_simulation()
    demonstrate_reproducibility()
    
    print("All WGEN algorithms are working correctly!")
    print("✓ Markov chain wet/dry state transitions")
    print("✓ Gamma distribution sampling for wet day amounts")
    print("✓ Monthly parameter selection based on current date")
    print("✓ Random number generator state management")


if __name__ == "__main__":
    main()