"""
Analytical engine for parameter extraction and trend analysis.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import json
from scipy import stats
from .base import Engine


@dataclass
class MonthlyParams:
    """Monthly precipitation parameters."""
    p_ww: float  # P(wet|wet)
    p_wd: float  # P(wet|dry) 
    alpha: float  # Gamma shape parameter
    beta: float   # Gamma scale parameter
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    seasonal_slopes: Dict[str, Dict[str, float]]  # season -> parameter -> slope
    significance_tests: Dict[str, Dict[str, float]]  # p-values
    trend_confidence: Dict[str, Dict[str, str]]  # significance levels
    regression_type: str = 'linear'  # Type of regression used
    validation_results: Optional[Dict[str, Dict[str, bool]]] = None  # Slope validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class WindowAnalysis:
    """Sliding window analysis results."""
    window_parameters: Dict[str, Dict[int, MonthlyParams]]  # window_id -> month -> params
    window_dates: Dict[str, tuple]  # window_id -> (start_date, end_date)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'window_parameters': {},
            'window_dates': {}
        }
        
        # Convert window parameters
        for window_id, month_params in self.window_parameters.items():
            result['window_parameters'][window_id] = {
                str(month): params.to_dict() for month, params in month_params.items()
            }
        
        # Convert window dates to ISO format strings
        for window_id, (start_date, end_date) in self.window_dates.items():
            result['window_dates'][window_id] = [
                start_date.isoformat(),
                end_date.isoformat()
            ]
        
        return result


@dataclass
class ParameterManifest:
    """Complete parameter manifest for simulation."""
    metadata: Dict[str, Any]
    overall_parameters: Dict[int, MonthlyParams]  # month -> parameters
    trend_analysis: Optional[TrendAnalysis]
    sliding_window_stats: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'metadata': self.metadata.copy(),
            'overall_parameters': {
                str(month): params.to_dict() 
                for month, params in self.overall_parameters.items()
            }
        }
        
        if self.trend_analysis is not None:
            result['trend_analysis'] = self.trend_analysis.to_dict()
        else:
            result['trend_analysis'] = None
            
        result['sliding_window_stats'] = self.sliding_window_stats
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save_to_file(self, filepath: str, indent: int = 2) -> None:
        """Save parameter manifest to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent, default=str)


class AnalyticalEngine(Engine):
    """
    Engine for extracting stochastic parameters and detecting trends.
    
    Implements Richardson & Wright (1984) parameter estimation with
    modern enhancements for trend detection and sliding window analysis.
    """
    
    def __init__(self, data: pd.Series, wet_day_threshold: float = 0.001):
        """
        Initialize analytical engine.
        
        Args:
            data: Time series of precipitation data
            wet_day_threshold: Minimum precipitation to classify as wet day (inches)
        """
        self.data = data.copy()
        self.wet_day_threshold = wet_day_threshold
        self.overall_parameters = None
        self.window_analysis = None
        self.trend_analysis = None
        
        # Convert threshold to mm if needed (assuming input is in inches)
        self.wet_day_threshold_mm = wet_day_threshold * 25.4
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize analytical engine.
        
        Args:
            **kwargs: Additional parameters (unused)
        """
        # Ensure data is sorted and clean
        self.data = self.data.sort_index()
        self.data = self.data.fillna(0.0)  # Fill missing with zero precipitation
    
    def reset(self, **kwargs) -> None:
        """
        Reset engine state.
        
        Args:
            **kwargs: Additional parameters (unused)
        """
        self.overall_parameters = None
        self.window_analysis = None
        self.trend_analysis = None
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current engine state.
        
        Returns:
            Dictionary containing current state
        """
        return {
            'data_length': len(self.data),
            'data_period': (self.data.index.min(), self.data.index.max()),
            'wet_day_threshold': self.wet_day_threshold,
            'has_overall_parameters': self.overall_parameters is not None,
            'has_window_analysis': self.window_analysis is not None,
            'has_trend_analysis': self.trend_analysis is not None
        }
    
    def calculate_monthly_parameters(self) -> Dict[int, MonthlyParams]:
        """
        Calculate overall monthly parameters for entire dataset.
        
        Returns:
            Dictionary mapping month number to parameters
        """
        monthly_params = {}
        
        for month in range(1, 13):
            month_data = self.data[self.data.index.month == month]
            
            if len(month_data) < 30:  # Need minimum data for reliable estimates
                # Use annual averages as fallback
                month_data = self.data
            
            params = self._calculate_parameters_for_period(month_data)
            monthly_params[month] = params
        
        self.overall_parameters = monthly_params
        return monthly_params
    
    def perform_sliding_window_analysis(self, window_years: int = 30) -> WindowAnalysis:
        """
        Perform sliding window analysis to evaluate parameter evolution.
        
        Args:
            window_years: Size of sliding window in years
            
        Returns:
            WindowAnalysis with windowed parameter estimates
        """
        if len(self.data) == 0:
            raise ValueError("No data available for sliding window analysis")
        
        # Store window size for later use in manifest generation
        self._window_years = window_years
        
        # Calculate window size in days
        window_days = window_years * 365
        
        # Get data date range
        start_date = self.data.index.min()
        end_date = self.data.index.max()
        total_days = (end_date - start_date).days
        
        if total_days < window_days:
            raise ValueError(f"Data period ({total_days} days) shorter than window size ({window_days} days)")
        
        window_parameters = {}
        window_dates = {}
        
        # Create overlapping windows (step by 1 year)
        step_days = 365
        window_id = 0
        
        current_start = start_date
        while current_start + pd.Timedelta(days=window_days) <= end_date:
            current_end = current_start + pd.Timedelta(days=window_days)
            
            # Extract window data
            window_data = self.data[
                (self.data.index >= current_start) & 
                (self.data.index <= current_end)
            ]
            
            if len(window_data) >= window_days * 0.8:  # Require 80% data availability
                # Calculate monthly parameters for this window
                window_monthly_params = {}
                for month in range(1, 13):
                    month_data = window_data[window_data.index.month == month]
                    if len(month_data) >= 10:  # Minimum monthly data
                        params = self._calculate_parameters_for_period(month_data)
                        window_monthly_params[month] = params
                
                if window_monthly_params:  # Only store if we have valid parameters
                    window_id_str = f"window_{window_id:03d}"
                    window_parameters[window_id_str] = window_monthly_params
                    window_dates[window_id_str] = (current_start, current_end)
                    window_id += 1
            
            current_start += pd.Timedelta(days=step_days)
        
        self.window_analysis = WindowAnalysis(
            window_parameters=window_parameters,
            window_dates=window_dates
        )
        
        return self.window_analysis
    
    def extract_trends(self, window_results: WindowAnalysis, 
                      regression_type: str = 'linear') -> TrendAnalysis:
        """
        Extract trends from sliding window analysis using linear or polynomial regression.
        
        Args:
            window_results: Results from sliding window analysis
            regression_type: Type of regression ('linear' or 'polynomial')
            
        Returns:
            TrendAnalysis with trend slopes and significance tests
        """
        if not window_results.window_parameters:
            raise ValueError("No window analysis results available for trend extraction")
        
        if regression_type not in ['linear', 'polynomial']:
            raise ValueError("regression_type must be 'linear' or 'polynomial'")
        
        # Organize data by season and parameter
        seasonal_data = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5], 
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_slopes = {}
        significance_tests = {}
        trend_confidence = {}
        
        for season, months in seasonal_data.items():
            seasonal_slopes[season] = {}
            significance_tests[season] = {}
            trend_confidence[season] = {}
            
            # Extract time series for each parameter
            param_names = ['p_ww', 'p_wd', 'alpha', 'beta']
            
            for param_name in param_names:
                param_values = []
                window_times = []
                
                for window_id, window_params in window_results.window_parameters.items():
                    # Average parameter across months in season
                    season_values = []
                    for month in months:
                        if month in window_params:
                            param_obj = window_params[month]
                            param_value = getattr(param_obj, param_name)
                            season_values.append(param_value)
                    
                    if season_values:
                        param_values.append(np.mean(season_values))
                        # Use window center time
                        start_date, end_date = window_results.window_dates[window_id]
                        center_time = start_date + (end_date - start_date) / 2
                        window_times.append(center_time.year + center_time.dayofyear / 365.25)
                
                if len(param_values) >= 3:  # Need minimum points for regression
                    slope, p_value = self._perform_regression_analysis(
                        window_times, param_values, regression_type
                    )
                    
                    seasonal_slopes[season][param_name] = slope
                    significance_tests[season][param_name] = p_value
                    
                    # Determine confidence level
                    confidence = self._determine_trend_confidence(p_value)
                    trend_confidence[season][param_name] = confidence
                else:
                    # Insufficient data
                    seasonal_slopes[season][param_name] = 0.0
                    significance_tests[season][param_name] = 1.0
                    trend_confidence[season][param_name] = "Insufficient data"
        
        self.trend_analysis = TrendAnalysis(
            seasonal_slopes=seasonal_slopes,
            significance_tests=significance_tests,
            trend_confidence=trend_confidence,
            regression_type=regression_type
        )
        
        # Validate trend slopes
        validation_results = self.validate_trend_slopes(self.trend_analysis)
        self.trend_analysis.validation_results = validation_results
        
        return self.trend_analysis
    
    def generate_parameter_manifest(self) -> ParameterManifest:
        """
        Generate complete parameter manifest for simulation.
        
        Returns:
            ParameterManifest with all analysis results
        """
        # Ensure we have overall parameters
        if self.overall_parameters is None:
            self.calculate_monthly_parameters()
        
        # Generate metadata
        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': [
                self.data.index.min().isoformat(),
                self.data.index.max().isoformat()
            ],
            'wet_day_threshold': self.wet_day_threshold,
            'data_completeness': (1 - self.data.isna().sum() / len(self.data)),
            'total_data_points': len(self.data)
        }
        
        # Calculate sliding window stats if available
        sliding_window_stats = None
        if self.window_analysis:
            sliding_window_stats = {
                'window_count': len(self.window_analysis.window_parameters),
                'window_years': getattr(self, '_window_years', 30),  # Use actual window size
                'analysis_method': 'overlapping_windows'
            }
            
            # Add trend analysis metadata if available
            if self.trend_analysis:
                sliding_window_stats.update({
                    'regression_type': self.trend_analysis.regression_type,
                    'trend_validation_performed': self.trend_analysis.validation_results is not None
                })
        
        return ParameterManifest(
            metadata=metadata,
            overall_parameters=self.overall_parameters,
            trend_analysis=self.trend_analysis,
            sliding_window_stats=sliding_window_stats
        )
    
    def perform_comprehensive_trend_analysis(self, window_years: int = 30, 
                                           regression_type: str = 'linear') -> TrendAnalysis:
        """
        Perform comprehensive trend analysis including sliding window analysis and trend extraction.
        
        Args:
            window_years: Size of sliding window in years
            regression_type: Type of regression ('linear' or 'polynomial')
            
        Returns:
            TrendAnalysis with complete trend information
        """
        # First perform sliding window analysis
        window_results = self.perform_sliding_window_analysis(window_years)
        
        # Then extract trends from the windowed results
        trend_results = self.extract_trends(window_results, regression_type)
        
        return trend_results
    
    def _calculate_parameters_for_period(self, data: pd.Series) -> MonthlyParams:
        """
        Calculate WGEN parameters for a specific time period.
        
        Args:
            data: Time series data for the period
            
        Returns:
            MonthlyParams with calculated parameters
        """
        if len(data) == 0:
            # Return default parameters for empty data
            return MonthlyParams(p_ww=0.5, p_wd=0.3, alpha=1.0, beta=5.0)
        
        # Classify wet/dry days
        wet_days = data > self.wet_day_threshold_mm
        
        # Calculate transition probabilities
        p_ww = self._calculate_transition_probability(wet_days, True, True)
        p_wd = self._calculate_transition_probability(wet_days, False, True)
        
        # Calculate Gamma parameters for wet days
        wet_amounts = data[wet_days]
        if len(wet_amounts) >= 2:
            alpha, beta = self._fit_gamma_parameters(wet_amounts)
        else:
            # Default parameters for insufficient wet days
            alpha, beta = 1.0, 5.0
        
        return MonthlyParams(p_ww=p_ww, p_wd=p_wd, alpha=alpha, beta=beta)
    
    def _calculate_transition_probability(self, wet_days: pd.Series, 
                                        from_state: bool, to_state: bool) -> float:
        """
        Calculate Markov chain transition probability.
        
        Args:
            wet_days: Boolean series indicating wet days
            from_state: Previous day state (True=wet, False=dry)
            to_state: Current day state (True=wet, False=dry)
            
        Returns:
            Transition probability
        """
        if len(wet_days) < 2:
            return 0.5  # Default probability
        
        # Find transitions
        prev_state = wet_days.shift(1)
        
        # Count transitions from specified state
        from_count = (prev_state == from_state).sum()
        
        if from_count == 0:
            return 0.5  # Default if no instances of from_state
        
        # Count transitions from_state -> to_state
        transition_count = ((prev_state == from_state) & (wet_days == to_state)).sum()
        
        return transition_count / from_count
    
    def _fit_gamma_parameters(self, wet_amounts: pd.Series) -> tuple:
        """
        Fit Gamma distribution parameters using method of moments.
        
        Args:
            wet_amounts: Precipitation amounts on wet days
            
        Returns:
            Tuple of (alpha, beta) parameters
        """
        if len(wet_amounts) == 0:
            return 1.0, 5.0
        
        # Method of moments estimation
        mean_precip = wet_amounts.mean()
        var_precip = wet_amounts.var()
        
        if var_precip <= 0 or mean_precip <= 0:
            return 1.0, 5.0
        
        # Gamma parameters: mean = alpha * beta, var = alpha * beta^2
        # Therefore: beta = var / mean, alpha = mean / beta
        beta = var_precip / mean_precip
        alpha = mean_precip / beta
        
        # Ensure parameters are positive and reasonable
        alpha = max(0.1, min(alpha, 10.0))
        beta = max(0.1, min(beta, 100.0))
        
        return alpha, beta
    
    def _perform_regression_analysis(self, x_values: list, y_values: list, 
                                   regression_type: str) -> tuple:
        """
        Perform regression analysis on parameter time series.
        
        Args:
            x_values: Time values (years)
            y_values: Parameter values
            regression_type: Type of regression ('linear' or 'polynomial')
            
        Returns:
            Tuple of (slope, p_value)
        """
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        if regression_type == 'linear':
            # Linear regression using scipy.stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
            return slope, p_value
            
        elif regression_type == 'polynomial':
            # Polynomial regression (degree 2) with significance testing
            if len(x_values) < 4:  # Need at least 4 points for quadratic
                # Fall back to linear
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
                return slope, p_value
            
            # Fit polynomial (degree 2)
            coeffs = np.polyfit(x_array, y_array, 2)
            poly_func = np.poly1d(coeffs)
            y_pred = poly_func(x_array)
            
            # Calculate R-squared and F-statistic for significance
            ss_res = np.sum((y_array - y_pred) ** 2)
            ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
            
            if ss_tot == 0:
                r_squared = 1.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            # F-test for polynomial regression significance
            n = len(y_array)
            p = 3  # Number of parameters (intercept + x + x^2)
            
            if n <= p or ss_res == 0:
                p_value = 0.001  # Assume significant if perfect fit
            else:
                f_stat = (r_squared / (p - 1)) / ((1 - r_squared) / (n - p))
                p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
            
            # Return linear component of trend (coefficient of x term)
            linear_slope = coeffs[1] if len(coeffs) > 1 else 0.0
            return linear_slope, p_value
        
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
    
    def _determine_trend_confidence(self, p_value: float) -> str:
        """
        Determine trend confidence level based on p-value.
        
        Args:
            p_value: Statistical significance p-value
            
        Returns:
            String describing confidence level
        """
        if p_value < 0.01:
            return "High (p < 0.01)"
        elif p_value < 0.05:
            return "Medium (p < 0.05)"
        elif p_value < 0.10:
            return "Low (p < 0.10)"
        else:
            return "Not significant"
    
    def validate_trend_slopes(self, trend_analysis: TrendAnalysis) -> Dict[str, Dict[str, bool]]:
        """
        Validate trend slopes to ensure they are within reasonable bounds.
        
        Args:
            trend_analysis: TrendAnalysis object to validate
            
        Returns:
            Dictionary indicating which slopes are valid
        """
        validation_results = {}
        
        # Define reasonable bounds for parameter changes per year
        bounds = {
            'p_ww': (-0.01, 0.01),  # Transition probabilities: ±1% per year max
            'p_wd': (-0.01, 0.01),
            'alpha': (-0.1, 0.1),   # Gamma parameters: more flexible bounds
            'beta': (-1.0, 1.0)
        }
        
        for season, season_slopes in trend_analysis.seasonal_slopes.items():
            validation_results[season] = {}
            
            for param_name, slope in season_slopes.items():
                if param_name in bounds:
                    min_bound, max_bound = bounds[param_name]
                    is_valid = min_bound <= slope <= max_bound
                    validation_results[season][param_name] = is_valid
                else:
                    validation_results[season][param_name] = True  # Unknown parameter, assume valid
        
        return validation_results
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with error handling and validation.
        
        Returns:
            Dictionary containing analysis report with error information
        """
        report = {
            'analysis_status': 'success',
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'analysis_results': {}
        }
        
        try:
            # Data quality assessment
            report['data_quality'] = {
                'total_data_points': len(self.data),
                'data_completeness': 1 - self.data.isna().sum() / len(self.data) if len(self.data) > 0 else 0,
                'data_period': [
                    self.data.index.min().isoformat() if len(self.data) > 0 else None,
                    self.data.index.max().isoformat() if len(self.data) > 0 else None
                ],
                'wet_day_fraction': (self.data > self.wet_day_threshold_mm).sum() / len(self.data) if len(self.data) > 0 else 0
            }
            
            # Check data quality
            if len(self.data) == 0:
                report['errors'].append("No data available for analysis")
                report['analysis_status'] = 'failed'
                return report
            
            if report['data_quality']['data_completeness'] < 0.8:
                report['warnings'].append(f"Low data completeness: {report['data_quality']['data_completeness']:.2%}")
            
            if report['data_quality']['wet_day_fraction'] < 0.05:
                report['warnings'].append(f"Very low wet day fraction: {report['data_quality']['wet_day_fraction']:.2%}")
            
            # Overall parameters analysis
            if self.overall_parameters is None:
                self.calculate_monthly_parameters()
            
            report['analysis_results']['overall_parameters_calculated'] = True
            report['analysis_results']['monthly_parameter_count'] = len(self.overall_parameters)
            
            # Sliding window analysis
            if self.window_analysis is not None:
                report['analysis_results']['sliding_window_analysis'] = {
                    'completed': True,
                    'window_count': len(self.window_analysis.window_parameters),
                    'window_years': getattr(self, '_window_years', 'unknown')
                }
            else:
                report['analysis_results']['sliding_window_analysis'] = {'completed': False}
            
            # Trend analysis
            if self.trend_analysis is not None:
                report['analysis_results']['trend_analysis'] = {
                    'completed': True,
                    'regression_type': self.trend_analysis.regression_type,
                    'seasons_analyzed': len(self.trend_analysis.seasonal_slopes),
                    'validation_performed': self.trend_analysis.validation_results is not None
                }
                
                # Check for invalid trend slopes
                if self.trend_analysis.validation_results:
                    invalid_slopes = []
                    for season, season_results in self.trend_analysis.validation_results.items():
                        for param, is_valid in season_results.items():
                            if not is_valid:
                                slope_value = self.trend_analysis.seasonal_slopes[season][param]
                                invalid_slopes.append(f"{season} {param}: {slope_value:.6f}")
                    
                    if invalid_slopes:
                        report['warnings'].append(f"Invalid trend slopes detected: {', '.join(invalid_slopes)}")
            else:
                report['analysis_results']['trend_analysis'] = {'completed': False}
            
        except Exception as e:
            report['analysis_status'] = 'failed'
            report['errors'].append(f"Analysis failed with error: {str(e)}")
        
        return report
    
    def export_results(self, output_dir: str, include_json: bool = True, 
                      include_report: bool = True) -> Dict[str, str]:
        """
        Export analysis results to files with comprehensive error handling.
        
        Args:
            output_dir: Directory to save output files
            include_json: Whether to save parameter manifest as JSON
            include_report: Whether to save analysis report
            
        Returns:
            Dictionary mapping output type to file path
        """
        import os
        from pathlib import Path
        
        output_files = {}
        
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate parameter manifest
            manifest = self.generate_parameter_manifest()
            
            if include_json:
                json_path = os.path.join(output_dir, 'parameter_manifest.json')
                manifest.save_to_file(json_path)
                output_files['parameter_manifest'] = json_path
            
            if include_report:
                report = self.generate_analysis_report()
                report_path = os.path.join(output_dir, 'analysis_report.json')
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                output_files['analysis_report'] = report_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to export results: {str(e)}")
        
        return output_files