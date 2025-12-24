"""
Monte Carlo precipitation simulation framework.

This module provides a high-level interface for running Monte Carlo
precipitation simulations with multiple methods and comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from ..data.ghcn_downloader import GHCNDownloader, find_nearby_stations
from ..data.ghcn_parser import GHCNParser
from .analysis import DroughtAnalyzer, ExtremeEventAnalyzer, SeasonalAnalyzer


class PrecipitationSimulator:
    """
    High-level precipitation simulator for water resources applications.
    
    This class provides a complete workflow for:
    - Loading historical precipitation data
    - Calculating simulation parameters
    - Running Monte Carlo simulations with multiple methods
    - Comprehensive analysis and visualization
    
    Example:
        >>> sim = pg.PrecipitationSimulator()
        >>> sim.load_historical_data(latitude=40.7608, longitude=-111.8910)
        >>> sim.calculate_monthly_parameters()
        >>> results = sim.run_monte_carlo_simulation(n_realizations=100)
        >>> analysis = sim.analyze_results()
    """
    
    def __init__(self, station_id: Optional[str] = None, cache_dir: str = 'data'):
        """
        Initialize the precipitation simulator.
        
        Args:
            station_id: Optional GHCN station ID. If None, will find nearest station.
            cache_dir: Directory for caching downloaded data
        """
        self.station_id = station_id
        self.cache_dir = cache_dir
        self.historical_data = None
        self.monthly_params = None
        self.results = {}
        self.station_info = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def load_historical_data(self, latitude: Optional[float] = None, 
                           longitude: Optional[float] = None,
                           radius_km: float = 50, min_years: int = 20) -> pd.Series:
        """
        Load and process historical precipitation data.
        
        Args:
            latitude: Latitude for station search (if station_id not provided)
            longitude: Longitude for station search (if station_id not provided)
            radius_km: Search radius in kilometers
            min_years: Minimum years of data required
            
        Returns:
            Historical precipitation time series
            
        Raises:
            ValueError: If no suitable stations found or data cannot be loaded
        """
        if self.station_id is None:
            if latitude is None or longitude is None:
                raise ValueError("Must provide either station_id or latitude/longitude")
            
            # Find nearest station
            stations = find_nearby_stations(
                latitude=latitude, longitude=longitude, 
                radius_km=radius_km, min_years=min_years
            )
            
            if not stations:
                raise ValueError(f"No suitable stations found within {radius_km} km")
            
            stations_df = pd.DataFrame(stations)
            self.station_id = stations_df.iloc[0]['id']
            self.station_info = stations_df.iloc[0].to_dict()
            
            self.logger.info(f"Using station: {self.station_id} - {self.station_info['name']}")
            self.logger.info(f"Distance: {self.station_info['distance_km']:.1f} km")
            self.logger.info(f"Data period: {self.station_info['first_year']}-{self.station_info['last_year']}")
        
        # Download and parse data
        downloader = GHCNDownloader(cache_dir=self.cache_dir)
        dly_path = downloader.download_station_data(self.station_id)
        
        parser = GHCNParser(dly_path)
        ghcn_data = parser.parse_dly_file(dly_path)
        self.historical_data = parser.extract_precipitation(ghcn_data).sort_index().astype(float)
        
        self.logger.info(f"Historical data loaded: {len(self.historical_data)} days")
        self.logger.info(f"Period: {self.historical_data.index.min().date()} to {self.historical_data.index.max().date()}")
        
        return self.historical_data
    
    def calculate_monthly_parameters(self, threshold: float = 0.1) -> pd.DataFrame:
        """
        Calculate monthly WGEN parameters from historical data.
        
        Args:
            threshold: Precipitation threshold for wet/dry classification (mm)
            
        Returns:
            DataFrame with monthly parameters
            
        Raises:
            ValueError: If historical data not loaded
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_historical_data() first.")
        
        rows = []
        for month in range(1, 13):
            monthly_data = self.historical_data[self.historical_data.index.month == month]
            
            if len(monthly_data) == 0:
                rows.append({
                    'month': month, 'p_w': np.nan, 'p_ww': np.nan, 'p_dw': np.nan,
                    'alpha': np.nan, 'beta': np.nan, 'mean_precip': np.nan,
                    'wet_days': 0, 'total_days': 0
                })
                continue
            
            # Wet/dry day analysis
            is_wet = (monthly_data > threshold).astype(int)
            p_w = is_wet.mean()
            
            # Transition probabilities
            if len(is_wet) > 1:
                wet_prev = is_wet.values[:-1]
                wet_next = is_wet.values[1:]
                
                p_ww = (np.sum((wet_prev == 1) & (wet_next == 1)) / np.sum(wet_prev == 1) 
                       if np.sum(wet_prev == 1) > 0 else np.nan)
                p_dw = (np.sum((wet_prev == 0) & (wet_next == 1)) / np.sum(wet_prev == 0) 
                       if np.sum(wet_prev == 0) > 0 else np.nan)
            else:
                p_ww = p_dw = np.nan
            
            # Gamma distribution parameters for wet days
            wet_amounts = monthly_data[monthly_data > threshold]
            if len(wet_amounts) > 1:
                mean_wet = wet_amounts.mean()
                var_wet = wet_amounts.var(ddof=1)
                
                if var_wet > 0 and mean_wet > 0:
                    alpha = (mean_wet ** 2) / var_wet
                    beta = var_wet / mean_wet
                else:
                    alpha = beta = np.nan
            else:
                alpha = beta = np.nan
            
            rows.append({
                'month': month,
                'p_w': p_w,
                'p_ww': p_ww,
                'p_dw': p_dw,
                'alpha': alpha,
                'beta': beta,
                'mean_precip': monthly_data.mean(),
                'wet_days': len(wet_amounts),
                'total_days': len(monthly_data)
            })
        
        self.monthly_params = pd.DataFrame(rows).set_index('month')
        return self.monthly_params
    
    def run_monte_carlo_simulation(self, 
                                 start_date: str = '2026-01-01', 
                                 end_date: str = '2035-12-31',
                                 n_realizations: int = 100, 
                                 methods: Optional[List[str]] = None, 
                                 seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Run Monte Carlo simulation with multiple methods.
        
        Args:
            start_date: Simulation start date (YYYY-MM-DD)
            end_date: Simulation end date (YYYY-MM-DD)
            n_realizations: Number of Monte Carlo realizations
            methods: List of simulation methods ('wgen', 'bootstrap_doy', 'block_bootstrap')
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of simulation results by method
            
        Raises:
            ValueError: If parameters not calculated or invalid method specified
        """
        if self.monthly_params is None:
            raise ValueError("Monthly parameters not calculated. Call calculate_monthly_parameters() first.")
        
        if methods is None:
            methods = ['wgen', 'bootstrap_doy', 'block_bootstrap']
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_years = len(dates) / 365.25
        
        self.logger.info(f"Running {n_realizations} realizations for {n_years:.1f} years ({len(dates)} days)")
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Running {method} simulation...")
            
            if method == 'wgen':
                sims = self._simulate_wgen(dates, n_realizations, seed)
            elif method == 'bootstrap_doy':
                sims = self._simulate_bootstrap_doy(dates, n_realizations, seed)
            elif method == 'block_bootstrap':
                sims = self._simulate_block_bootstrap(dates, n_realizations, seed)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[method] = sims
        
        self.results = results
        return results
    
    def _simulate_wgen(self, dates: pd.DatetimeIndex, n_realizations: int, seed: int) -> pd.DataFrame:
        """WGEN-style simulation using monthly parameters."""
        rng = np.random.default_rng(seed)
        sims = np.zeros((len(dates), n_realizations))
        
        for real in range(n_realizations):
            prev_wet = None
            
            for i, date in enumerate(dates):
                month = date.month
                params = self.monthly_params.loc[month]
                
                # Determine if day is wet
                if prev_wet is None:
                    prob_wet = params['p_w'] if pd.notna(params['p_w']) else 0.1
                else:
                    if prev_wet:
                        prob_wet = params['p_ww'] if pd.notna(params['p_ww']) else params['p_w']
                    else:
                        prob_wet = params['p_dw'] if pd.notna(params['p_dw']) else params['p_w']
                    
                    if pd.isna(prob_wet):
                        prob_wet = 0.1
                
                is_wet = rng.random() < prob_wet
                
                # Generate precipitation amount
                if is_wet:
                    alpha, beta = params['alpha'], params['beta']
                    if pd.notna(alpha) and pd.notna(beta) and alpha > 0 and beta > 0:
                        amount = rng.gamma(alpha, beta)
                    else:
                        # Fallback to historical sampling
                        monthly_wet = self.historical_data[
                            (self.historical_data.index.month == month) & 
                            (self.historical_data > 0.1)
                        ]
                        if len(monthly_wet) > 0:
                            amount = float(rng.choice(monthly_wet.values))
                        else:
                            amount = 0.0
                else:
                    amount = 0.0
                
                sims[i, real] = amount
                prev_wet = is_wet
        
        return pd.DataFrame(sims, index=dates)
    
    def _simulate_bootstrap_doy(self, dates: pd.DatetimeIndex, n_realizations: int, seed: int) -> pd.DataFrame:
        """Day-of-year bootstrap simulation."""
        rng = np.random.default_rng(seed)
        
        # Create pools by day of year
        doy_pools = {}
        for date, value in self.historical_data.dropna().items():
            doy = (date.month, date.day)
            if doy not in doy_pools:
                doy_pools[doy] = []
            doy_pools[doy].append(value)
        
        sims = np.zeros((len(dates), n_realizations))
        
        for i, date in enumerate(dates):
            doy = (date.month, date.day)
            
            # Get pool for this day of year, fallback to month if needed
            if doy in doy_pools:
                pool = doy_pools[doy]
            else:
                # Fallback to all days in the month
                pool = []
                for key, values in doy_pools.items():
                    if key[0] == date.month:
                        pool.extend(values)
                
                if not pool:
                    pool = list(self.historical_data.dropna().values)
            
            # Sample from pool
            sims[i, :] = rng.choice(pool, size=n_realizations, replace=True)
        
        return pd.DataFrame(sims, index=dates)
    
    def _simulate_block_bootstrap(self, dates: pd.DatetimeIndex, n_realizations: int, 
                                seed: int, block_length: int = 7) -> pd.DataFrame:
        """Block bootstrap simulation preserving temporal correlation."""
        rng = np.random.default_rng(seed)
        historical_values = self.historical_data.dropna().values
        n_historical = len(historical_values)
        
        sims = np.zeros((len(dates), n_realizations))
        
        for real in range(n_realizations):
            simulated = []
            
            while len(simulated) < len(dates):
                # Random starting position
                start_idx = rng.integers(0, max(1, n_historical - block_length + 1))
                block = historical_values[start_idx:start_idx + block_length]
                simulated.extend(block)
            
            sims[:, real] = simulated[:len(dates)]
        
        return pd.DataFrame(sims, index=dates)
    
    def analyze_results(self, save_plots: bool = True, output_dir: str = 'output') -> Dict:
        """
        Comprehensive analysis of simulation results.
        
        Args:
            save_plots: Whether to save analysis plots
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            ValueError: If no simulation results available
        """
        if not self.results:
            raise ValueError("No simulation results available. Run run_monte_carlo_simulation() first.")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        analysis = {}
        
        # Historical statistics for comparison
        hist_annual = self.historical_data.resample('YE').sum()
        
        self.logger.info("=== SIMULATION ANALYSIS ===")
        self.logger.info(f"Historical annual mean: {hist_annual.mean():.1f} mm")
        self.logger.info(f"Historical annual std: {hist_annual.std():.1f} mm")
        
        for method, sims in self.results.items():
            self.logger.info(f"\n--- {method.upper()} METHOD ---")
            
            # Annual statistics
            annual_sims = sims.resample('YE').sum()
            annual_means = annual_sims.mean(axis=1)
            annual_stds = annual_sims.std(axis=1)
            
            self.logger.info(f"Simulated annual mean: {annual_means.mean():.1f} ± {annual_means.std():.1f} mm")
            self.logger.info(f"Simulated annual std: {annual_stds.mean():.1f} ± {annual_stds.std():.1f} mm")
            
            # Monthly statistics
            monthly_sims = sims.resample('ME').sum()
            
            # Wet day statistics
            wet_day_freq = (sims > 0.1).mean(axis=1).resample('YE').mean()
            self.logger.info(f"Wet day frequency: {wet_day_freq.mean():.3f} ± {wet_day_freq.std():.3f}")
            
            # Extreme events (95th percentile)
            daily_95th = sims[sims > 0].quantile(0.95, axis=1)
            self.logger.info(f"95th percentile daily: {daily_95th.mean():.1f} ± {daily_95th.std():.1f} mm")
            
            analysis[method] = {
                'annual_totals': annual_sims,
                'monthly_totals': monthly_sims,
                'annual_means': annual_means,
                'annual_stds': annual_stds,
                'wet_day_freq': wet_day_freq,
                'daily_95th': daily_95th
            }
        
        # Create comprehensive plots
        if save_plots:
            self._create_analysis_plots(analysis, output_dir)
        
        return analysis
    
    def _create_analysis_plots(self, analysis: Dict, output_dir: str):
        """Create comprehensive analysis plots."""
        
        # 1. Annual precipitation comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Historical annual totals
        hist_annual = self.historical_data.resample('YE').sum()
        
        # Annual totals comparison
        ax = axes[0, 0]
        methods = list(analysis.keys())
        annual_data = [analysis[method]['annual_totals'].values.flatten() for method in methods]
        
        ax.boxplot([hist_annual.values] + annual_data, 
                  labels=['Historical'] + [m.title() for m in methods])
        ax.set_ylabel('Annual Precipitation (mm)')
        ax.set_title('Annual Precipitation Distribution')
        ax.grid(True, alpha=0.3)
        
        # Monthly climatology
        ax = axes[0, 1]
        hist_monthly_clim = self.historical_data.groupby(self.historical_data.index.month).mean()
        
        months = range(1, 13)
        ax.plot(months, hist_monthly_clim.values, 'ko-', label='Historical', linewidth=2)
        
        for method in methods:
            monthly_sims = analysis[method]['monthly_totals']
            monthly_clim = monthly_sims.groupby(monthly_sims.index.month).mean().mean(axis=1)
            ax.plot(months, monthly_clim.values, 'o-', label=method.title(), alpha=0.7)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Monthly Precipitation (mm)')
        ax.set_title('Monthly Climatology')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Wet day frequency by month
        ax = axes[1, 0]
        hist_wet_freq = (self.historical_data > 0.1).groupby(self.historical_data.index.month).mean()
        ax.plot(months, hist_wet_freq.values, 'ko-', label='Historical', linewidth=2)
        
        for method in methods:
            sims = self.results[method]
            wet_freq = (sims > 0.1).groupby(sims.index.month).mean().mean(axis=1)
            ax.plot(months, wet_freq.values, 'o-', label=method.title(), alpha=0.7)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Wet Day Frequency')
        ax.set_title('Seasonal Wet Day Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annual time series (first 10 realizations)
        ax = axes[1, 1]
        years = analysis[methods[0]]['annual_totals'].index.year
        
        for i in range(min(10, analysis[methods[0]]['annual_totals'].shape[1])):
            ax.plot(years, analysis[methods[0]]['annual_totals'].iloc[:, i], 
                   alpha=0.3, color='blue')
        
        ax.plot(hist_annual.index.year, hist_annual.values, 'ko-', 
               label='Historical', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Precipitation (mm)')
        ax.set_title(f'{methods[0].title()} - Sample Realizations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precipitation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Analysis plots saved to {output_dir}/")
    
    def export_results(self, output_dir: str = 'output', 
                      include_sample_realizations: bool = True,
                      n_sample_realizations: int = 10) -> None:
        """
        Export simulation results and analysis to files.
        
        Args:
            output_dir: Directory to save outputs
            include_sample_realizations: Whether to save sample realizations
            n_sample_realizations: Number of sample realizations to save
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save parameters
        if self.monthly_params is not None:
            self.monthly_params.to_csv(f'{output_dir}/monthly_parameters.csv')
        
        # Save results
        for method, sims in self.results.items():
            # Save sample realizations
            if include_sample_realizations:
                sample_sims = sims.iloc[:, :n_sample_realizations]
                sample_sims.to_csv(f'{output_dir}/{method}_sample_realizations.csv')
            
            # Save annual statistics
            annual_stats = sims.resample('YE').sum()
            annual_summary = pd.DataFrame({
                'mean': annual_stats.mean(axis=1),
                'std': annual_stats.std(axis=1),
                'min': annual_stats.min(axis=1),
                'max': annual_stats.max(axis=1),
                'p10': annual_stats.quantile(0.1, axis=1),
                'p90': annual_stats.quantile(0.9, axis=1)
            })
            annual_summary.to_csv(f'{output_dir}/{method}_annual_statistics.csv')
        
        self.logger.info(f"Results exported to {output_dir}/")
    
    def get_water_resources_analysis(self) -> Dict:
        """
        Get specialized water resources analysis.
        
        Returns:
            Dictionary with drought, extreme event, and seasonal analysis
        """
        if not self.results:
            raise ValueError("No simulation results available")
        
        analysis = {}
        
        for method, sims in self.results.items():
            # Drought analysis
            drought_analyzer = DroughtAnalyzer()
            drought_stats = drought_analyzer.analyze(sims)
            
            # Extreme event analysis
            extreme_analyzer = ExtremeEventAnalyzer()
            extreme_stats = extreme_analyzer.analyze(sims)
            
            # Seasonal analysis
            seasonal_analyzer = SeasonalAnalyzer()
            seasonal_stats = seasonal_analyzer.analyze(sims)
            
            analysis[method] = {
                'drought': drought_stats,
                'extreme_events': extreme_stats,
                'seasonal': seasonal_stats
            }
        
        return analysis