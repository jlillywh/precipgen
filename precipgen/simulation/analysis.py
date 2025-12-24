"""
Analysis modules for precipitation simulation results.

This module provides specialized analyzers for water resources applications
including drought analysis, extreme events, and seasonal patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class DroughtAnalyzer:
    """Analyzer for drought characteristics in precipitation simulations."""
    
    def __init__(self, threshold_percentile: float = 20):
        """
        Initialize drought analyzer.
        
        Args:
            threshold_percentile: Percentile below which conditions are considered drought
        """
        self.threshold_percentile = threshold_percentile
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, precip_data: pd.DataFrame) -> Dict:
        """
        Analyze drought characteristics from precipitation data.
        
        Args:
            precip_data: DataFrame with precipitation simulations
            
        Returns:
            Dictionary with drought statistics
        """
        # Calculate monthly totals
        monthly_totals = precip_data.resample('ME').sum()
        
        # Define drought threshold
        threshold = monthly_totals.quantile(self.threshold_percentile/100, axis=1)
        
        drought_stats = {}
        
        for col in monthly_totals.columns:
            monthly_series = monthly_totals[col]
            
            # Identify drought months
            drought_months = monthly_series < threshold
            
            # Calculate drought characteristics
            drought_events = self._identify_drought_events(drought_months, threshold, monthly_series)
            
            if drought_events:
                durations = [d['duration'] for d in drought_events]
                severities = [d['severity'] for d in drought_events]
                
                drought_stats[col] = {
                    'n_events': len(drought_events),
                    'mean_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'mean_severity': np.mean(severities),
                    'max_severity': np.max(severities),
                    'events': drought_events
                }
            else:
                drought_stats[col] = {
                    'n_events': 0,
                    'mean_duration': 0,
                    'max_duration': 0,
                    'mean_severity': 0,
                    'max_severity': 0,
                    'events': []
                }
        
        # Summarize across realizations
        summary = self._summarize_drought_stats(drought_stats)
        
        return {
            'individual_realizations': drought_stats,
            'summary': summary
        }
    
    def _identify_drought_events(self, drought_months: pd.Series, 
                               threshold: pd.Series, 
                               monthly_series: pd.Series) -> List[Dict]:
        """Identify individual drought events."""
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
        
        return drought_events
    
    def _summarize_drought_stats(self, drought_stats: Dict) -> Dict:
        """Summarize drought statistics across realizations."""
        n_events = [stats['n_events'] for stats in drought_stats.values()]
        mean_durations = [stats['mean_duration'] for stats in drought_stats.values() if stats['n_events'] > 0]
        max_durations = [stats['max_duration'] for stats in drought_stats.values() if stats['n_events'] > 0]
        
        summary = {
            'events_per_period': {
                'mean': np.mean(n_events),
                'std': np.std(n_events),
                'min': np.min(n_events),
                'max': np.max(n_events)
            }
        }
        
        if mean_durations:
            summary['duration_months'] = {
                'mean': np.mean(mean_durations),
                'std': np.std(mean_durations),
                'max_observed': np.max(max_durations)
            }
        
        return summary


class ExtremeEventAnalyzer:
    """Analyzer for extreme precipitation events and return periods."""
    
    def __init__(self, return_periods: Optional[List[int]] = None):
        """
        Initialize extreme event analyzer.
        
        Args:
            return_periods: List of return periods to calculate (years)
        """
        self.return_periods = return_periods or [2, 5, 10, 25, 50, 100]
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, precip_data: pd.DataFrame) -> Dict:
        """
        Analyze extreme precipitation events and return periods.
        
        Args:
            precip_data: DataFrame with daily precipitation simulations
            
        Returns:
            Dictionary with extreme event statistics
        """
        # Annual maximum daily precipitation
        annual_max = precip_data.resample('YE').max()
        
        extreme_stats = {}
        
        for col in annual_max.columns:
            annual_max_series = annual_max[col].dropna()
            
            if len(annual_max_series) > 0:
                # Calculate return levels
                return_levels = self._calculate_return_levels(annual_max_series)
                
                extreme_stats[col] = {
                    'annual_max_mean': annual_max_series.mean(),
                    'annual_max_std': annual_max_series.std(),
                    'return_levels': return_levels
                }
        
        # Summarize across realizations
        summary = self._summarize_extreme_stats(extreme_stats)
        
        return {
            'individual_realizations': extreme_stats,
            'summary': summary
        }
    
    def _calculate_return_levels(self, annual_max_series: pd.Series) -> Dict:
        """Calculate return levels using empirical method."""
        # Sort in descending order
        sorted_max = np.sort(annual_max_series.values)[::-1]
        
        # Calculate return period estimates using plotting position
        n = len(sorted_max)
        plotting_positions = [(i + 1) / (n + 1) for i in range(n)]
        return_period_empirical = [1 / p for p in plotting_positions]
        
        # Interpolate for desired return periods
        return_levels = {}
        for rp in self.return_periods:
            if rp <= max(return_period_empirical):
                return_levels[rp] = np.interp(rp, return_period_empirical[::-1], sorted_max[::-1])
            else:
                # Mark as extrapolated
                return_levels[rp] = np.nan
        
        return return_levels
    
    def _summarize_extreme_stats(self, extreme_stats: Dict) -> Dict:
        """Summarize extreme event statistics across realizations."""
        summary = {}
        
        # Summarize return levels
        for rp in self.return_periods:
            return_levels = [stats['return_levels'].get(rp, np.nan) 
                           for stats in extreme_stats.values()]
            return_levels = [rl for rl in return_levels if not np.isnan(rl)]
            
            if return_levels:
                summary[f'{rp}_year_return_level'] = {
                    'mean': np.mean(return_levels),
                    'std': np.std(return_levels),
                    'min': np.min(return_levels),
                    'max': np.max(return_levels)
                }
        
        return summary


class SeasonalAnalyzer:
    """Analyzer for seasonal precipitation patterns and water availability."""
    
    def __init__(self, seasons: Optional[Dict[str, List[int]]] = None):
        """
        Initialize seasonal analyzer.
        
        Args:
            seasons: Dictionary defining seasons (default: standard 4 seasons)
        """
        self.seasons = seasons or {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5], 
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, precip_data: pd.DataFrame) -> Dict:
        """
        Analyze seasonal water availability patterns.
        
        Args:
            precip_data: DataFrame with daily precipitation simulations
            
        Returns:
            Dictionary with seasonal statistics
        """
        seasonal_stats = {}
        
        for season_name, months in self.seasons.items():
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
                },
                'annual_totals': seasonal_totals
            }
        
        return seasonal_stats
    
    def get_monthly_climatology(self, precip_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly climatology from simulation data.
        
        Args:
            precip_data: DataFrame with daily precipitation simulations
            
        Returns:
            DataFrame with monthly statistics
        """
        monthly_stats = []
        
        for month in range(1, 13):
            monthly_data = precip_data[precip_data.index.month == month]
            
            if len(monthly_data) > 0:
                monthly_totals = monthly_data.resample('ME').sum()
                
                stats = {
                    'month': month,
                    'mean': monthly_totals.mean().mean(),
                    'std': monthly_totals.std().mean(),
                    'p10': monthly_totals.quantile(0.1, axis=1).mean(),
                    'p25': monthly_totals.quantile(0.25, axis=1).mean(),
                    'p50': monthly_totals.quantile(0.5, axis=1).mean(),
                    'p75': monthly_totals.quantile(0.75, axis=1).mean(),
                    'p90': monthly_totals.quantile(0.9, axis=1).mean()
                }
                monthly_stats.append(stats)
        
        return pd.DataFrame(monthly_stats).set_index('month')