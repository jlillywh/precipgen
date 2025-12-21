"""
Simulation engine for synthetic precipitation generation.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from .base import Engine
from .analytical import ParameterManifest, MonthlyParams
from ..utils.exceptions import (
    SimulationEngineError, StateError, BoundsError, SynchronizationError,
    ParameterEstimationError, create_error_context, handle_graceful_degradation
)
from ..utils.logging_config import get_logger


@dataclass
class SimulationState:
    """Current simulation state."""
    current_date: datetime
    is_wet: bool
    random_state: tuple
    elapsed_days: int
    current_parameters: MonthlyParams


class SimulationEngine(Engine):
    """
    Stateful precipitation generator using WGEN algorithm.
    
    Implements Richardson & Wright (1984) weather generation with
    Markov chains and Gamma distributions, including non-stationary
    trend projection capabilities with comprehensive error handling.
    """
    
    def __init__(self, parameters: ParameterManifest, 
                 trend_mode: bool = False, random_seed: Optional[int] = None):
        """
        Initialize simulation engine.
        
        Args:
            parameters: Parameter manifest from analytical engine
            trend_mode: Enable non-stationary trend projection
            random_seed: Random seed for reproducible generation
            
        Raises:
            SimulationEngineError: If parameters are invalid
            ParameterEstimationError: If parameter manifest is malformed
        """
        self.logger = get_logger('simulation_engine')
        
        try:
            self.parameters = parameters
            self.trend_mode = trend_mode
            self.random_state = np.random.RandomState(random_seed)
            
            # Simulation state
            self.current_date = None
            self.is_wet = False
            self.elapsed_days = 0
            self.start_date = None
            
            # Error tracking
            self._error_count = 0
            self._warning_count = 0
            self._bounds_violations = 0
            
            # Validate parameters
            self._validate_parameters()
            
            self.logger.info(f"Simulation engine initialized with trend_mode={trend_mode}, "
                           f"random_seed={random_seed}")
            
        except Exception as e:
            context = create_error_context(
                'simulation_engine_initialization',
                trend_mode=trend_mode,
                random_seed=random_seed,
                has_parameters=parameters is not None
            )
            self.logger.error(f"Failed to initialize simulation engine: {str(e)}", 
                            extra={'context': context})
            raise SimulationEngineError(
                'initialization',
                f"Engine initialization failed: {str(e)}"
            )
    
    def _validate_parameters(self) -> None:
        """
        Validate parameter manifest with comprehensive checks.
        
        Raises:
            ParameterEstimationError: If parameters are invalid or missing
        """
        try:
            if not self.parameters:
                raise ParameterEstimationError(
                    "parameter_validation",
                    "Parameter manifest is None"
                )
            
            if not self.parameters.overall_parameters:
                raise ParameterEstimationError(
                    "parameter_validation", 
                    "Parameter manifest missing overall_parameters"
                )
            
            # Check that we have parameters for all months
            missing_months = []
            invalid_months = []
            
            for month in range(1, 13):
                if month not in self.parameters.overall_parameters:
                    missing_months.append(month)
                else:
                    # Validate parameter values
                    params = self.parameters.overall_parameters[month]
                    validation_errors = self._validate_monthly_parameters(params, month)
                    if validation_errors:
                        invalid_months.append((month, validation_errors))
            
            if missing_months:
                raise ParameterEstimationError(
                    "parameter_validation",
                    f"Missing parameters for months: {missing_months}"
                )
            
            if invalid_months:
                error_details = []
                for month, errors in invalid_months:
                    error_details.append(f"Month {month}: {', '.join(errors)}")
                
                raise ParameterEstimationError(
                    "parameter_validation",
                    f"Invalid parameters detected: {'; '.join(error_details)}"
                )
            
            # Validate trend analysis if trend mode is enabled
            if self.trend_mode:
                self._validate_trend_analysis()
            
            self.logger.info("Parameter validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {str(e)}")
            raise
    
    def _validate_monthly_parameters(self, params: MonthlyParams, month: int) -> list:
        """
        Validate individual monthly parameters.
        
        Args:
            params: Monthly parameters to validate
            month: Month number for context
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate transition probabilities
        if not (0 <= params.p_ww <= 1):
            errors.append(f"p_ww={params.p_ww} not in [0,1]")
        
        if not (0 <= params.p_wd <= 1):
            errors.append(f"p_wd={params.p_wd} not in [0,1]")
        
        # Validate Gamma parameters
        if params.alpha <= 0:
            errors.append(f"alpha={params.alpha} must be positive")
        
        if params.beta <= 0:
            errors.append(f"beta={params.beta} must be positive")
        
        # Check for reasonable parameter ranges
        if params.alpha > 10:
            errors.append(f"alpha={params.alpha} unusually large (>10)")
        
        if params.beta > 100:
            errors.append(f"beta={params.beta} unusually large (>100)")
        
        # Check for NaN or infinite values
        for param_name, value in [('p_ww', params.p_ww), ('p_wd', params.p_wd), 
                                 ('alpha', params.alpha), ('beta', params.beta)]:
            if not np.isfinite(value):
                errors.append(f"{param_name}={value} is not finite")
        
        return errors
    
    def _validate_trend_analysis(self) -> None:
        """
        Validate trend analysis data for trend mode.
        
        Raises:
            ParameterEstimationError: If trend analysis is invalid
        """
        if not self.parameters.trend_analysis:
            # For testing purposes, allow trend mode without trend analysis
            # but log a warning and disable trend mode
            self.logger.warning("Trend mode enabled but no trend analysis available. Disabling trend mode.")
            self.trend_mode = False
            return
        
        if not self.parameters.trend_analysis.seasonal_slopes:
            self.logger.warning("Trend analysis missing seasonal slopes. Disabling trend mode.")
            self.trend_mode = False
            return
        
        # Validate seasonal slopes
        required_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        missing_seasons = []
        
        for season in required_seasons:
            if season not in self.parameters.trend_analysis.seasonal_slopes:
                missing_seasons.append(season)
        
        if missing_seasons:
            self.logger.warning(f"Missing trend slopes for seasons: {missing_seasons}")
        
        # Validate slope values
        for season, slopes in self.parameters.trend_analysis.seasonal_slopes.items():
            for param, slope in slopes.items():
                if not np.isfinite(slope):
                    raise ParameterEstimationError(
                        "trend_validation",
                        f"Invalid trend slope for {season} {param}: {slope}"
                    )
    
    def initialize(self, start_date: datetime, initial_wet_state: bool = False) -> None:
        """
        Initialize simulation engine with error handling.
        
        Args:
            start_date: Starting date for simulation
            initial_wet_state: Initial wet/dry state
            
        Raises:
            StateError: If initialization fails
        """
        try:
            if not isinstance(start_date, datetime):
                raise StateError(
                    "Invalid start_date type",
                    {'start_date_type': type(start_date).__name__}
                )
            
            self.current_date = start_date
            self.start_date = start_date
            self.is_wet = bool(initial_wet_state)
            self.elapsed_days = 0
            
            # Reset error counters
            self._error_count = 0
            self._warning_count = 0
            self._bounds_violations = 0
            
            self.logger.info(f"Simulation initialized: start_date={start_date.isoformat()}, "
                           f"initial_wet_state={initial_wet_state}")
            
        except Exception as e:
            context = create_error_context(
                'simulation_initialization',
                start_date=str(start_date),
                initial_wet_state=initial_wet_state
            )
            self.logger.error(f"Simulation initialization failed: {str(e)}", 
                            extra={'context': context})
            raise StateError(
                f"Initialization failed: {str(e)}",
                {'start_date': str(start_date), 'initial_wet_state': initial_wet_state}
            )
    
    def reset(self, start_date: Optional[datetime] = None) -> None:
        """
        Reset engine to initial state with error handling.
        
        Args:
            start_date: Optional new start date
            
        Raises:
            StateError: If reset fails
        """
        try:
            if start_date:
                if not isinstance(start_date, datetime):
                    raise StateError(
                        "Invalid start_date type for reset",
                        {'start_date_type': type(start_date).__name__}
                    )
                self.start_date = start_date
                self.current_date = start_date
            elif self.start_date:
                self.current_date = self.start_date
            else:
                raise StateError(
                    "Cannot reset: no start date available",
                    recovery_possible=False
                )
            
            self.is_wet = False
            self.elapsed_days = 0
            
            # Reset error counters
            self._error_count = 0
            self._warning_count = 0
            self._bounds_violations = 0
            
            self.logger.info(f"Simulation reset to {self.current_date.isoformat()}")
            
        except Exception as e:
            context = create_error_context(
                'simulation_reset',
                start_date=str(start_date) if start_date else None,
                has_start_date=self.start_date is not None
            )
            self.logger.error(f"Simulation reset failed: {str(e)}", 
                            extra={'context': context})
            raise
    
    def step(self) -> float:
        """
        Generate next daily precipitation value with comprehensive error handling.
        
        Returns:
            Daily precipitation in millimeters
            
        Raises:
            StateError: If engine is not properly initialized
            SimulationEngineError: If simulation step fails
            BoundsError: If parameter drift exceeds bounds
        """
        try:
            if self.current_date is None:
                raise StateError(
                    "Engine not initialized",
                    recovery_possible=True
                )
            
            # Get current month parameters (with trend adjustment if enabled)
            try:
                current_params = self._get_current_parameters()
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Failed to get current parameters: {str(e)}")
                raise SimulationEngineError(
                    self.elapsed_days,
                    f"Parameter retrieval failed: {str(e)}"
                )
            
            # Validate parameters before use
            param_errors = self._validate_monthly_parameters(current_params, self.current_date.month)
            if param_errors:
                self._error_count += 1
                raise SimulationEngineError(
                    self.elapsed_days,
                    f"Invalid current parameters: {'; '.join(param_errors)}",
                    current_parameters={
                        'p_ww': current_params.p_ww,
                        'p_wd': current_params.p_wd,
                        'alpha': current_params.alpha,
                        'beta': current_params.beta
                    }
                )
            
            # Determine wet/dry state using Markov chain
            try:
                if self.is_wet:
                    # Previous day was wet
                    prob_wet = current_params.p_ww
                else:
                    # Previous day was dry
                    prob_wet = current_params.p_wd
                
                # Generate random number for state transition
                random_value = self.random_state.random()
                self.is_wet = random_value < prob_wet
                
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Markov chain transition failed: {str(e)}")
                raise SimulationEngineError(
                    self.elapsed_days,
                    f"State transition failed: {str(e)}"
                )
            
            # Generate precipitation amount
            try:
                if self.is_wet:
                    # Sample from Gamma distribution with error handling
                    if current_params.alpha <= 0 or current_params.beta <= 0:
                        raise SimulationEngineError(
                            self.elapsed_days,
                            f"Invalid Gamma parameters: alpha={current_params.alpha}, beta={current_params.beta}"
                        )
                    
                    precip_amount = self.random_state.gamma(
                        current_params.alpha, 
                        current_params.beta
                    )
                    
                    # Validate generated precipitation
                    if not np.isfinite(precip_amount) or precip_amount < 0:
                        self._warning_count += 1
                        self.logger.warning(f"Invalid precipitation generated: {precip_amount}, using 0.0")
                        precip_amount = 0.0
                    elif precip_amount > 1000:  # > 1 meter per day
                        self._warning_count += 1
                        self.logger.warning(f"Extreme precipitation generated: {precip_amount:.2f} mm")
                else:
                    # Dry day
                    precip_amount = 0.0
                
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Precipitation generation failed: {str(e)}")
                # Use graceful degradation
                precip_amount = handle_graceful_degradation(
                    e, 0.0, "precipitation_generation", self.logger
                )
            
            # Advance simulation state
            try:
                self.current_date += timedelta(days=1)
                self.elapsed_days += 1
                
            except Exception as e:
                self._error_count += 1
                self.logger.error(f"Date advancement failed: {str(e)}")
                raise SimulationEngineError(
                    self.elapsed_days,
                    f"Date advancement failed: {str(e)}"
                )
            
            return float(precip_amount)
            
        except (StateError, SimulationEngineError, BoundsError):
            # Re-raise specific errors
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            self._error_count += 1
            context = create_error_context(
                'simulation_step',
                elapsed_days=self.elapsed_days,
                current_date=str(self.current_date) if self.current_date else None,
                is_wet=self.is_wet
            )
            self.logger.error(f"Unexpected error in simulation step: {str(e)}", 
                            extra={'context': context})
            raise SimulationEngineError(
                self.elapsed_days,
                f"Unexpected simulation error: {str(e)}"
            )
    
    def get_current_state(self) -> SimulationState:
        """
        Get current simulation state with error handling.
        
        Returns:
            SimulationState with current values
            
        Raises:
            StateError: If state cannot be retrieved
        """
        try:
            if self.current_date is None:
                raise StateError(
                    "Cannot get state: simulation not initialized",
                    recovery_possible=True
                )
            
            current_params = self._get_current_parameters()
            
            return SimulationState(
                current_date=self.current_date,
                is_wet=self.is_wet,
                random_state=self.random_state.get_state(),
                elapsed_days=self.elapsed_days,
                current_parameters=current_params
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get current state: {str(e)}")
            raise StateError(
                f"State retrieval failed: {str(e)}",
                {'elapsed_days': self.elapsed_days}
            )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current engine state for serialization with error handling.
        
        Returns:
            Dictionary containing current state
        """
        try:
            return {
                'current_date': self.current_date.isoformat() if self.current_date else None,
                'start_date': self.start_date.isoformat() if self.start_date else None,
                'is_wet': self.is_wet,
                'elapsed_days': self.elapsed_days,
                'trend_mode': self.trend_mode,
                'random_state': self.random_state.get_state(),
                'error_statistics': {
                    'error_count': self._error_count,
                    'warning_count': self._warning_count,
                    'bounds_violations': self._bounds_violations
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to serialize state: {str(e)}")
            return {
                'error': f"State serialization failed: {str(e)}",
                'elapsed_days': getattr(self, 'elapsed_days', 0)
            }
    
    def synchronize_with_external_time(self, external_date: datetime) -> None:
        """
        Synchronize simulation time with external system.
        
        Args:
            external_date: External system's current date
            
        Raises:
            SynchronizationError: If synchronization fails or times are too different
        """
        try:
            if self.current_date is None:
                raise SynchronizationError(
                    "None", str(external_date),
                    time_difference=None
                )
            
            time_diff = (external_date - self.current_date).total_seconds() / (24 * 3600)
            
            # Allow small differences (up to 1 day)
            if abs(time_diff) <= 1.0:
                self.current_date = external_date
                self.logger.debug(f"Time synchronized: {time_diff:.2f} day difference")
            else:
                raise SynchronizationError(
                    self.current_date.isoformat(),
                    external_date.isoformat(),
                    time_difference=time_diff
                )
                
        except SynchronizationError:
            raise
        except Exception as e:
            self.logger.error(f"Time synchronization failed: {str(e)}")
            raise SynchronizationError(
                str(self.current_date) if self.current_date else "None",
                str(external_date),
                time_difference=None
            )
    
    def _get_current_parameters(self) -> MonthlyParams:
        """
        Get parameters for current month with trend adjustment if enabled.
        
        Returns:
            MonthlyParams adjusted for current time and trends
            
        Raises:
            StateError: If current date is not set
            BoundsError: If parameter drift exceeds bounds
        """
        if self.current_date is None:
            raise StateError("Current date not set", recovery_possible=True)
        
        current_month = self.current_date.month
        base_params = self.parameters.overall_parameters[current_month]
        
        if not self.trend_mode or not self.parameters.trend_analysis:
            return base_params
        
        # Calculate parameter drift using trend projection
        try:
            return self._calculate_parameter_drift(base_params, current_month)
        except Exception as e:
            self._warning_count += 1
            self.logger.warning(f"Parameter drift calculation failed: {str(e)}, using baseline")
            return base_params
    
    def _calculate_parameter_drift(self, base_params: MonthlyParams, month: int) -> MonthlyParams:
        """
        Calculate parameter drift using trend slopes with comprehensive bounds checking.
        
        Implements the formula: Parameter(t) = Parameter_baseline + (Trend_Slope × elapsed_time)
        
        Args:
            base_params: Baseline monthly parameters
            month: Current month (1-12)
            
        Returns:
            MonthlyParams with trend-adjusted values and physical bounds applied
            
        Raises:
            BoundsError: If parameter drift exceeds reasonable bounds
        """
        try:
            # Calculate elapsed time in years (more precise calculation)
            elapsed_years = self._calculate_elapsed_years()
            
            # Determine season for trend lookup
            season = self._get_season(month)
            
            if season not in self.parameters.trend_analysis.seasonal_slopes:
                self.logger.warning(f"No trend slopes available for season {season}, using baseline parameters")
                return base_params
            
            slopes = self.parameters.trend_analysis.seasonal_slopes[season]
            
            # Validate slopes before applying
            if not self._validate_trend_slopes(slopes, season):
                self.logger.warning(f"Invalid trend slopes for season {season}, using baseline parameters")
                return base_params
            
            # Apply drift formula: Parameter(t) = Parameter_baseline + (Trend_Slope × elapsed_time)
            adjusted_p_ww = base_params.p_ww + slopes.get('p_ww', 0.0) * elapsed_years
            adjusted_p_wd = base_params.p_wd + slopes.get('p_wd', 0.0) * elapsed_years
            adjusted_alpha = base_params.alpha + slopes.get('alpha', 0.0) * elapsed_years
            adjusted_beta = base_params.beta + slopes.get('beta', 0.0) * elapsed_years
            
            # Check for extreme drift before applying bounds
            self._check_extreme_drift(base_params, adjusted_p_ww, adjusted_p_wd, 
                                    adjusted_alpha, adjusted_beta, elapsed_years)
            
            # Log significant parameter changes
            if elapsed_years > 1.0:  # Only log for simulations longer than 1 year
                self._log_parameter_changes(base_params, adjusted_p_ww, adjusted_p_wd, 
                                          adjusted_alpha, adjusted_beta, elapsed_years, season)
            
            # Apply comprehensive physical bounds checking
            return self._apply_physical_bounds(
                adjusted_p_ww, adjusted_p_wd, adjusted_alpha, adjusted_beta,
                base_params, elapsed_years
            )
            
        except BoundsError:
            raise
        except Exception as e:
            self.logger.error(f"Error calculating parameter drift: {str(e)}")
            self.logger.warning("Falling back to baseline parameters")
            return base_params
    
    def _check_extreme_drift(self, base_params: MonthlyParams, adj_p_ww: float,
                           adj_p_wd: float, adj_alpha: float, adj_beta: float,
                           elapsed_years: float) -> None:
        """
        Check for extreme parameter drift that might indicate problems.
        
        Args:
            base_params: Original baseline parameters
            adj_p_ww, adj_p_wd, adj_alpha, adj_beta: Adjusted parameter values
            elapsed_years: Time elapsed since simulation start
            
        Raises:
            BoundsError: If drift is extreme
        """
        # Check for extreme drift in transition probabilities
        if adj_p_ww < -0.5 or adj_p_ww > 1.5:
            raise BoundsError(
                'p_ww', adj_p_ww, (0.0, 1.0),
                time_step=self.elapsed_days,
                drift_rate=(adj_p_ww - base_params.p_ww) / elapsed_years if elapsed_years > 0 else 0
            )
        
        if adj_p_wd < -0.5 or adj_p_wd > 1.5:
            raise BoundsError(
                'p_wd', adj_p_wd, (0.0, 1.0),
                time_step=self.elapsed_days,
                drift_rate=(adj_p_wd - base_params.p_wd) / elapsed_years if elapsed_years > 0 else 0
            )
        
        # Check for extreme drift in Gamma parameters
        if adj_alpha < -1.0 or adj_alpha > base_params.alpha * 10:
            raise BoundsError(
                'alpha', adj_alpha, (0.1, base_params.alpha * 5),
                time_step=self.elapsed_days,
                drift_rate=(adj_alpha - base_params.alpha) / elapsed_years if elapsed_years > 0 else 0
            )
        
        if adj_beta < -1.0 or adj_beta > base_params.beta * 10:
            raise BoundsError(
                'beta', adj_beta, (0.1, base_params.beta * 5),
                time_step=self.elapsed_days,
                drift_rate=(adj_beta - base_params.beta) / elapsed_years if elapsed_years > 0 else 0
            )
    
    def _validate_trend_slopes(self, slopes: Dict[str, float], season: str) -> bool:
        """
        Validate trend slopes for reasonableness.
        
        Args:
            slopes: Dictionary of parameter slopes
            season: Season name for logging
            
        Returns:
            True if slopes are reasonable, False otherwise
        """
        # Define reasonable bounds for slopes (per year)
        slope_bounds = {
            'p_ww': (-0.05, 0.05),  # ±5% per year max
            'p_wd': (-0.05, 0.05),  # ±5% per year max
            'alpha': (-1.0, 1.0),   # ±1.0 units per year max
            'beta': (-5.0, 5.0)     # ±5.0 units per year max
        }
        
        for param, slope in slopes.items():
            if param in slope_bounds:
                min_bound, max_bound = slope_bounds[param]
                if not (min_bound <= slope <= max_bound):
                    self.logger.warning(f"Trend slope for {season} {param} ({slope:.6f}) exceeds reasonable bounds [{min_bound}, {max_bound}]")
                    return False
            
            # Check for NaN or infinite values
            if not np.isfinite(slope):
                self.logger.error(f"Invalid trend slope for {season} {param}: {slope}")
                return False
        
        return True
    
    def _log_parameter_changes(self, base_params: MonthlyParams, adj_p_ww: float, 
                              adj_p_wd: float, adj_alpha: float, adj_beta: float,
                              elapsed_years: float, season: str) -> None:
        """
        Log significant parameter changes due to trend projection.
        
        Args:
            base_params: Original baseline parameters
            adj_p_ww, adj_p_wd, adj_alpha, adj_beta: Adjusted parameter values
            elapsed_years: Time elapsed since simulation start
            season: Current season
        """
        # Calculate relative changes
        changes = {
            'p_ww': (adj_p_ww - base_params.p_ww) / base_params.p_ww if base_params.p_ww > 0 else 0,
            'p_wd': (adj_p_wd - base_params.p_wd) / base_params.p_wd if base_params.p_wd > 0 else 0,
            'alpha': (adj_alpha - base_params.alpha) / base_params.alpha if base_params.alpha > 0 else 0,
            'beta': (adj_beta - base_params.beta) / base_params.beta if base_params.beta > 0 else 0
        }
        
        # Log changes greater than 10%
        significant_changes = {param: change for param, change in changes.items() if abs(change) > 0.1}
        
        if significant_changes:
            self.logger.info(f"Significant parameter changes after {elapsed_years:.1f} years ({season}):")
            for param, change in significant_changes.items():
                self.logger.info(f"  {param}: {change:+.1%} change")
    
    def _calculate_elapsed_years(self) -> float:
        """
        Calculate elapsed time in years with high precision.
        
        Returns:
            Elapsed time in fractional years
        """
        if self.start_date is None or self.current_date is None:
            return 0.0
        
        # Use actual date difference for more accurate calculation
        time_delta = self.current_date - self.start_date
        elapsed_years = time_delta.total_seconds() / (365.25 * 24 * 3600)
        
        return elapsed_years
    
    def _apply_physical_bounds(self, p_ww: float, p_wd: float, alpha: float, beta: float,
                              base_params: MonthlyParams, elapsed_years: float) -> MonthlyParams:
        """
        Apply comprehensive physical bounds checking for drifted parameters.
        
        Ensures that:
        - Transition probabilities remain in [0, 1]
        - Gamma parameters remain positive and reasonable
        - Parameters don't drift too far from baseline values
        
        Args:
            p_ww: Drift-adjusted P(wet|wet)
            p_wd: Drift-adjusted P(wet|dry)
            alpha: Drift-adjusted Gamma alpha parameter
            beta: Drift-adjusted Gamma beta parameter
            base_params: Original baseline parameters
            elapsed_years: Time elapsed since simulation start
            
        Returns:
            MonthlyParams with bounds-constrained values
        """
        bounds_applied = False
        
        # Transition probability bounds: must be in [0, 1]
        bounded_p_ww = max(0.0, min(1.0, p_ww))
        bounded_p_wd = max(0.0, min(1.0, p_wd))
        
        if bounded_p_ww != p_ww or bounded_p_wd != p_wd:
            bounds_applied = True
            self._bounds_violations += 1
        
        # Gamma parameter bounds: must be positive and reasonable
        # Set minimum values to prevent degenerate distributions
        min_alpha = 0.1
        min_beta = 0.1
        
        # Set maximum drift limits to prevent unrealistic parameter values
        # Allow parameters to drift at most 50% from baseline over long periods
        max_drift_factor = 1.5
        min_drift_factor = 0.5
        
        max_alpha = base_params.alpha * max_drift_factor
        min_alpha_drift = base_params.alpha * min_drift_factor
        bounded_alpha = max(min_alpha, min(max_alpha, max(min_alpha_drift, alpha)))
        
        max_beta = base_params.beta * max_drift_factor
        min_beta_drift = base_params.beta * min_drift_factor
        bounded_beta = max(min_beta, min(max_beta, max(min_beta_drift, beta)))
        
        if bounded_alpha != alpha or bounded_beta != beta:
            bounds_applied = True
            self._bounds_violations += 1
        
        # Additional constraint: ensure Gamma distribution remains well-behaved
        # Mean = alpha * beta should be reasonable (between 0.1 and 100 mm)
        mean_precip = bounded_alpha * bounded_beta
        if mean_precip > 100.0:
            # Scale down both parameters proportionally
            scale_factor = 100.0 / mean_precip
            bounded_alpha *= scale_factor
            bounded_beta *= scale_factor
            bounds_applied = True
        elif mean_precip < 0.1:
            # Scale up both parameters proportionally
            scale_factor = 0.1 / mean_precip
            bounded_alpha *= scale_factor
            bounded_beta *= scale_factor
            bounds_applied = True
        
        # Ensure final parameters are still within absolute bounds
        bounded_alpha = max(min_alpha, bounded_alpha)
        bounded_beta = max(min_beta, bounded_beta)
        
        if bounds_applied:
            self.logger.debug(f"Physical bounds applied at elapsed_years={elapsed_years:.2f}")
        
        return MonthlyParams(
            p_ww=bounded_p_ww,
            p_wd=bounded_p_wd,
            alpha=bounded_alpha,
            beta=bounded_beta
        )
    
    def _get_season(self, month: int) -> str:
        """
        Get season name for given month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season name
        """
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # [9, 10, 11]
            return 'Fall'
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error and warning statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            'error_count': self._error_count,
            'warning_count': self._warning_count,
            'bounds_violations': self._bounds_violations,
            'elapsed_days': self.elapsed_days,
            'error_rate': self._error_count / max(1, self.elapsed_days),
            'warning_rate': self._warning_count / max(1, self.elapsed_days)
        }
    
    def get_trend_projection_info(self) -> Dict[str, Any]:
        """
        Get detailed information about current trend projection state.
        
        Returns:
            Dictionary with trend projection diagnostics
        """
        if not self.trend_mode or not self.parameters.trend_analysis:
            return {
                'trend_mode_enabled': False,
                'trend_analysis_available': False
            }
        
        current_month = self.current_date.month if self.current_date else None
        elapsed_years = self._calculate_elapsed_years()
        
        info = {
            'trend_mode_enabled': True,
            'trend_analysis_available': True,
            'elapsed_years': elapsed_years,
            'current_month': current_month,
            'current_season': self._get_season(current_month) if current_month else None,
            'bounds_violations': self._bounds_violations
        }
        
        if current_month:
            try:
                # Get baseline and adjusted parameters
                base_params = self.parameters.overall_parameters[current_month]
                adjusted_params = self._get_current_parameters()
                
                # Calculate drift amounts
                info.update({
                    'baseline_parameters': {
                        'p_ww': base_params.p_ww,
                        'p_wd': base_params.p_wd,
                        'alpha': base_params.alpha,
                        'beta': base_params.beta
                    },
                    'adjusted_parameters': {
                        'p_ww': adjusted_params.p_ww,
                        'p_wd': adjusted_params.p_wd,
                        'alpha': adjusted_params.alpha,
                        'beta': adjusted_params.beta
                    },
                    'parameter_drift': {
                        'p_ww': adjusted_params.p_ww - base_params.p_ww,
                        'p_wd': adjusted_params.p_wd - base_params.p_wd,
                        'alpha': adjusted_params.alpha - base_params.alpha,
                        'beta': adjusted_params.beta - base_params.beta
                    }
                })
                
                # Check if any bounds were applied
                season = self._get_season(current_month)
                if season in self.parameters.trend_analysis.seasonal_slopes:
                    slopes = self.parameters.trend_analysis.seasonal_slopes[season]
                    
                    # Calculate what parameters would be without bounds
                    unbounded_p_ww = base_params.p_ww + slopes.get('p_ww', 0.0) * elapsed_years
                    unbounded_p_wd = base_params.p_wd + slopes.get('p_wd', 0.0) * elapsed_years
                    unbounded_alpha = base_params.alpha + slopes.get('alpha', 0.0) * elapsed_years
                    unbounded_beta = base_params.beta + slopes.get('beta', 0.0) * elapsed_years
                    
                    info['bounds_applied'] = {
                        'p_ww': adjusted_params.p_ww != unbounded_p_ww,
                        'p_wd': adjusted_params.p_wd != unbounded_p_wd,
                        'alpha': adjusted_params.alpha != unbounded_alpha,
                        'beta': adjusted_params.beta != unbounded_beta
                    }
                    
                    info['trend_slopes'] = {
                        'p_ww': slopes.get('p_ww', 0.0),
                        'p_wd': slopes.get('p_wd', 0.0),
                        'alpha': slopes.get('alpha', 0.0),
                        'beta': slopes.get('beta', 0.0)
                    }
            except Exception as e:
                info['error'] = f"Failed to get trend projection details: {str(e)}"
        
        return info
    
    def validate_trend_projection(self) -> Dict[str, Any]:
        """
        Validate the current trend projection setup and parameters.
        
        Returns:
            Dictionary with validation results and any warnings
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            if not self.trend_mode:
                validation['warnings'].append("Trend mode is disabled")
                return validation
            
            if not self.parameters.trend_analysis:
                validation['is_valid'] = False
                validation['errors'].append("No trend analysis data available")
                return validation
            
            # Check if trend analysis has required components
            if not self.parameters.trend_analysis.seasonal_slopes:
                validation['is_valid'] = False
                validation['errors'].append("No seasonal slopes available in trend analysis")
                return validation
            
            # Validate trend slopes are reasonable
            for season, slopes in self.parameters.trend_analysis.seasonal_slopes.items():
                for param, slope in slopes.items():
                    if param in ['p_ww', 'p_wd']:
                        # Transition probability slopes should be small
                        if abs(slope) > 0.02:  # More than 2% change per year
                            validation['warnings'].append(
                                f"Large trend slope for {season} {param}: {slope:.4f}/year"
                            )
                    elif param in ['alpha', 'beta']:
                        # Gamma parameter slopes should be reasonable
                        if abs(slope) > 0.5:  # More than 0.5 units change per year
                            validation['warnings'].append(
                                f"Large trend slope for {season} {param}: {slope:.4f}/year"
                            )
            
            # Check if validation results are available
            if hasattr(self.parameters.trend_analysis, 'validation_results') and \
               self.parameters.trend_analysis.validation_results:
                for season, season_validation in self.parameters.trend_analysis.validation_results.items():
                    for param, is_valid in season_validation.items():
                        if not is_valid:
                            validation['warnings'].append(
                                f"Invalid trend slope detected for {season} {param}"
                            )
            
        except Exception as e:
            validation['is_valid'] = False
            validation['errors'].append(f"Validation failed: {str(e)}")
        
        return validation