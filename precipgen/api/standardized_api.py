"""
Standardized API interface for PrecipGen library.

Provides a unified, standardized interface that uses only standard Python
data types for seamless integration with external simulation environments.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

from ..engines.bootstrap import BootstrapEngine
from ..engines.analytical import AnalyticalEngine
from ..engines.simulation import SimulationEngine
from ..config.precipgen_config import PrecipGenConfig
from ..data.ghcn_parser import GHCNParser
from ..data.validator import DataValidator

from .data_exchange import DataExchangeFormat, SimulationState, ParameterSet, ValidationReport
from .synchronization import ExternalSimulationSync

logger = logging.getLogger(__name__)


class StandardizedAPI:
    """
    Standardized API interface for PrecipGen library.
    
    Provides a unified interface using only standard Python data types
    (dictionaries, lists, numpy arrays) for external integration.
    All methods return standard Python data structures for maximum compatibility.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], PrecipGenConfig]] = None):
        """
        Initialize standardized API.
        
        Args:
            config: Configuration dictionary or PrecipGenConfig instance
        """
        # Convert config to standard format
        if isinstance(config, PrecipGenConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = PrecipGenConfig(config_dict=config)
        elif config is None:
            # Create a minimal default configuration that allows empty data sources for testing
            self.config = PrecipGenConfig(config_dict={})
        else:
            self.config = PrecipGenConfig()
        
        # Initialize components
        self.engines: Dict[str, Any] = {}
        self.sync_manager = ExternalSimulationSync()
        self.data_exchange = DataExchangeFormat()
        
        # State tracking
        self.current_session_id: Optional[str] = None
        self.session_metadata: Dict[str, Any] = {}
    
    # Configuration Management
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration as standard dictionary.
        
        Returns:
            Configuration dictionary with standard Python data types
        """
        return self.data_exchange.to_standard_dict(self.config)
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            Validation report as dictionary
        """
        try:
            # Update configuration
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Validate updated configuration
            validation_errors = self.config.validate()
            
            report = ValidationReport()
            if validation_errors:
                for error in validation_errors:
                    report.add_error(error)
            else:
                report.add_info("Configuration updated successfully")
            
            return report.to_dict()
            
        except Exception as e:
            report = ValidationReport()
            report.add_error(f"Configuration update failed: {str(e)}")
            return report.to_dict()
    
    # Engine Management
    def create_bootstrap_engine(self, 
                               historical_data: Union[List[float], np.ndarray, Dict[str, Any]],
                               mode: str = 'random',
                               **kwargs) -> Dict[str, Any]:
        """
        Create bootstrap engine with standardized interface.
        
        Args:
            historical_data: Historical precipitation data as list, array, or time series dict
            mode: Sampling mode ('random' or 'sequential')
            **kwargs: Additional engine parameters
            
        Returns:
            Engine creation result as dictionary
        """
        try:
            # Convert input data to pandas Series with proper datetime index
            if isinstance(historical_data, pd.Series):
                # If it's already a Series, use it directly
                data_series = historical_data.copy()
                # If it doesn't have a datetime index, create one
                if not isinstance(data_series.index, pd.DatetimeIndex):
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            elif isinstance(historical_data, dict) and 'values' in historical_data:
                # Time series format
                data_series = pd.Series(historical_data['values'])
                if 'dates' in historical_data:
                    dates = pd.to_datetime(historical_data['dates'])
                    data_series.index = dates
                else:
                    # Create a datetime index if dates not provided
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            else:
                # Simple array format - create datetime index
                data_array = self.data_exchange.to_numpy_array(historical_data)
                start_date = pd.Timestamp('2000-01-01')
                data_series = pd.Series(data_array, 
                                      index=pd.date_range(start_date, periods=len(data_array), freq='D'))
            
            # Create engine
            engine = BootstrapEngine(data_series, mode=mode, **kwargs)
            engine_id = f"bootstrap_{len(self.engines)}"
            self.engines[engine_id] = engine
            
            result = {
                'success': True,
                'engine_id': engine_id,
                'engine_type': 'bootstrap',
                'mode': mode,
                'data_length': len(data_series),
                'message': f"Bootstrap engine created successfully with {len(data_series)} data points"
            }
            
            logger.info(f"Bootstrap engine created: {engine_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create bootstrap engine: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'engine_type': 'bootstrap'
            }
    
    def create_analytical_engine(self, 
                                data: Union[List[float], np.ndarray, Dict[str, Any]],
                                wet_day_threshold: float = 0.001,
                                **kwargs) -> Dict[str, Any]:
        """
        Create analytical engine with standardized interface.
        
        Args:
            data: Precipitation data as list, array, or time series dict
            wet_day_threshold: Minimum precipitation for wet day classification
            **kwargs: Additional engine parameters
            
        Returns:
            Engine creation result as dictionary
        """
        try:
            # Convert input data to pandas Series with proper datetime index
            if isinstance(data, pd.Series):
                # If it's already a Series, use it directly
                data_series = data.copy()
                # If it doesn't have a datetime index, create one
                if not isinstance(data_series.index, pd.DatetimeIndex):
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            elif isinstance(data, dict) and 'values' in data:
                data_series = pd.Series(data['values'])
                if 'dates' in data:
                    dates = pd.to_datetime(data['dates'])
                    data_series.index = dates
                else:
                    # Create a datetime index if dates not provided
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            else:
                # Convert to numpy array then Series with datetime index
                data_array = self.data_exchange.to_numpy_array(data)
                start_date = pd.Timestamp('2000-01-01')
                data_series = pd.Series(data_array, 
                                      index=pd.date_range(start_date, periods=len(data_array), freq='D'))
            
            # Create engine
            engine = AnalyticalEngine(data_series, wet_day_threshold=wet_day_threshold, **kwargs)
            engine_id = f"analytical_{len(self.engines)}"
            self.engines[engine_id] = engine
            
            result = {
                'success': True,
                'engine_id': engine_id,
                'engine_type': 'analytical',
                'wet_day_threshold': wet_day_threshold,
                'data_length': len(data_series),
                'message': f"Analytical engine created successfully with {len(data_series)} data points"
            }
            
            logger.info(f"Analytical engine created: {engine_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create analytical engine: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'engine_type': 'analytical'
            }
    
    def get_engine_info(self, engine_id: str) -> Dict[str, Any]:
        """
        Get engine information as standard dictionary.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Engine information dictionary
        """
        if engine_id not in self.engines:
            return {
                'success': False,
                'error': f"Engine not found: {engine_id}"
            }
        
        engine = self.engines[engine_id]
        
        try:
            info = {
                'success': True,
                'engine_id': engine_id,
                'engine_type': engine.__class__.__name__.lower().replace('engine', ''),
                'has_state_method': hasattr(engine, 'get_state'),
                'has_step_method': hasattr(engine, 'step'),
                'has_reset_method': hasattr(engine, 'reset')
            }
            
            # Add engine-specific information
            if hasattr(engine, 'get_state'):
                try:
                    state = engine.get_state()
                    info['current_state'] = self.data_exchange.to_standard_dict(state)
                except Exception as e:
                    info['state_error'] = str(e)
            
            return info
            
        except Exception as e:
            return {
                'success': False,
                'engine_id': engine_id,
                'error': str(e)
            }
    
    def list_engines(self) -> Dict[str, Any]:
        """
        List all registered engines.
        
        Returns:
            Dictionary with engine list and summary information
        """
        engines_info = {}
        
        for engine_id, engine in self.engines.items():
            engines_info[engine_id] = {
                'engine_type': engine.__class__.__name__.lower().replace('engine', ''),
                'has_state': hasattr(engine, 'get_state'),
                'has_step': hasattr(engine, 'step')
            }
        
        return {
            'engines': engines_info,
            'total_count': len(self.engines),
            'engine_types': list(set(info['engine_type'] for info in engines_info.values()))
        }
    
    # Simulation Operations
    def run_simulation_step(self, engine_id: str) -> Dict[str, Any]:
        """
        Run single simulation step for specified engine.
        
        Args:
            engine_id: Engine identifier
            
        Returns:
            Simulation step result as dictionary
        """
        if engine_id not in self.engines:
            return {
                'success': False,
                'error': f"Engine not found: {engine_id}"
            }
        
        engine = self.engines[engine_id]
        
        try:
            if not hasattr(engine, 'step'):
                return {
                    'success': False,
                    'error': f"Engine {engine_id} does not support step operation"
                }
            
            # Run simulation step
            result_value = engine.step()
            
            # Get current state if available
            current_state = None
            if hasattr(engine, 'get_state'):
                try:
                    state = engine.get_state()
                    current_state = self.data_exchange.to_standard_dict(state)
                except Exception as e:
                    logger.warning(f"Could not get state from engine {engine_id}: {str(e)}")
            
            # Update synchronization if enabled
            if self.sync_manager.sync_enabled and hasattr(engine, 'current_date'):
                self.sync_manager.advance_internal_clock(timedelta(days=1))
            
            return {
                'success': True,
                'engine_id': engine_id,
                'value': float(result_value),
                'current_state': current_state,
                'data_type': 'precipitation_mm'
            }
            
        except Exception as e:
            logger.error(f"Simulation step failed for engine {engine_id}: {str(e)}")
            return {
                'success': False,
                'engine_id': engine_id,
                'error': str(e)
            }
    
    def run_simulation_batch(self, 
                            engine_id: str, 
                            num_steps: int,
                            return_states: bool = False) -> Dict[str, Any]:
        """
        Run multiple simulation steps for specified engine.
        
        Args:
            engine_id: Engine identifier
            num_steps: Number of simulation steps to run
            return_states: Whether to return intermediate states
            
        Returns:
            Batch simulation result as dictionary
        """
        if engine_id not in self.engines:
            return {
                'success': False,
                'error': f"Engine not found: {engine_id}"
            }
        
        engine = self.engines[engine_id]
        
        try:
            if not hasattr(engine, 'step'):
                return {
                    'success': False,
                    'error': f"Engine {engine_id} does not support step operation"
                }
            
            values = []
            states = [] if return_states else None
            
            for i in range(num_steps):
                # Run step
                value = engine.step()
                values.append(float(value))
                
                # Collect state if requested
                if return_states and hasattr(engine, 'get_state'):
                    try:
                        state = engine.get_state()
                        states.append(self.data_exchange.to_standard_dict(state))
                    except Exception as e:
                        states.append({'error': str(e)})
                
                # Update synchronization
                if self.sync_manager.sync_enabled and hasattr(engine, 'current_date'):
                    self.sync_manager.advance_internal_clock(timedelta(days=1))
            
            result = {
                'success': True,
                'engine_id': engine_id,
                'values': values,
                'num_steps': num_steps,
                'data_type': 'precipitation_mm_series'
            }
            
            if return_states and states:
                result['states'] = states
            
            return result
            
        except Exception as e:
            logger.error(f"Batch simulation failed for engine {engine_id}: {str(e)}")
            return {
                'success': False,
                'engine_id': engine_id,
                'error': str(e),
                'completed_steps': len(values) if 'values' in locals() else 0
            }
    
    # Analysis Operations
    def analyze_data(self, 
                    data: Union[List[float], np.ndarray, Dict[str, Any]],
                    analysis_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis.
        
        Args:
            data: Precipitation data to analyze
            analysis_config: Analysis configuration parameters
            
        Returns:
            Analysis results as dictionary
        """
        try:
            # Convert data to pandas Series with proper datetime index
            if isinstance(data, pd.Series):
                # If it's already a Series, use it directly
                data_series = data.copy()
                # If it doesn't have a datetime index, create one
                if not isinstance(data_series.index, pd.DatetimeIndex):
                    # Create a simple datetime index starting from 2000-01-01
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            elif isinstance(data, dict) and 'values' in data:
                data_series = pd.Series(data['values'])
                if 'dates' in data:
                    data_series.index = pd.to_datetime(data['dates'])
                else:
                    # Create a datetime index if dates not provided
                    start_date = pd.Timestamp('2000-01-01')
                    data_series.index = pd.date_range(start_date, periods=len(data_series), freq='D')
            else:
                # Convert to numpy array then Series with datetime index
                data_array = self.data_exchange.to_numpy_array(data)
                start_date = pd.Timestamp('2000-01-01')
                data_series = pd.Series(data_array, 
                                      index=pd.date_range(start_date, periods=len(data_array), freq='D'))
            
            # Set up analysis configuration
            config = analysis_config or {}
            wet_day_threshold = config.get('wet_day_threshold', 0.001)
            window_years = config.get('window_years', 30)
            
            # Create analytical engine
            engine = AnalyticalEngine(data_series, wet_day_threshold=wet_day_threshold)
            
            # Perform analysis
            monthly_params = engine.calculate_monthly_parameters()
            window_analysis = engine.perform_sliding_window_analysis(window_years=window_years)
            trend_analysis = engine.extract_trends(window_analysis)
            parameter_manifest = engine.generate_parameter_manifest()
            
            # Convert results to standard format
            result = {
                'success': True,
                'analysis_config': {
                    'wet_day_threshold': wet_day_threshold,
                    'window_years': window_years,
                    'data_length': len(data_series)
                },
                'monthly_parameters': self.data_exchange.to_standard_dict(monthly_params),
                'trend_analysis': self.data_exchange.to_standard_dict(trend_analysis),
                'parameter_manifest': self.data_exchange.to_standard_dict(parameter_manifest)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Synchronization Management
    def enable_external_sync(self, 
                            external_start_time: Union[str, datetime],
                            internal_start_time: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Enable synchronization with external simulation.
        
        Args:
            external_start_time: External simulation start time (ISO string or datetime)
            internal_start_time: Internal start time (defaults to external)
            
        Returns:
            Synchronization setup result
        """
        try:
            # Convert time inputs
            if isinstance(external_start_time, str):
                external_time = datetime.fromisoformat(external_start_time)
            else:
                external_time = external_start_time
            
            if internal_start_time:
                if isinstance(internal_start_time, str):
                    internal_time = datetime.fromisoformat(internal_start_time)
                else:
                    internal_time = internal_start_time
            else:
                internal_time = None
            
            # Enable synchronization
            self.sync_manager.enable_synchronization(external_time, internal_time)
            
            return {
                'success': True,
                'external_start_time': external_time.isoformat(),
                'internal_start_time': (internal_time or external_time).isoformat(),
                'sync_enabled': True,
                'message': "External synchronization enabled"
            }
            
        except Exception as e:
            logger.error(f"Failed to enable external sync: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_external_time(self, external_time: Union[str, datetime]) -> Dict[str, Any]:
        """
        Update external simulation time.
        
        Args:
            external_time: New external time (ISO string or datetime)
            
        Returns:
            Synchronization update result
        """
        try:
            # Convert time input
            if isinstance(external_time, str):
                ext_time = datetime.fromisoformat(external_time)
            else:
                ext_time = external_time
            
            # Update synchronization
            sync_maintained = self.sync_manager.update_external_clock(ext_time)
            
            return {
                'success': True,
                'external_time': ext_time.isoformat(),
                'sync_maintained': sync_maintained,
                'sync_state': self.sync_manager.get_sync_state()
            }
            
        except Exception as e:
            logger.error(f"Failed to update external time: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status.
        
        Returns:
            Synchronization status dictionary
        """
        return self.sync_manager.get_sync_state()
    
    # Session Management
    def start_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a new simulation session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
            
        Returns:
            Session start result
        """
        try:
            self.current_session_id = session_id
            self.session_metadata = metadata or {}
            self.session_metadata['start_time'] = datetime.now().isoformat()
            self.session_metadata['engine_count'] = len(self.engines)
            
            logger.info(f"Session started: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'start_time': self.session_metadata['start_time'],
                'metadata': self.session_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def end_session(self) -> Dict[str, Any]:
        """
        End current simulation session.
        
        Returns:
            Session end result
        """
        try:
            if self.current_session_id is None:
                return {
                    'success': False,
                    'error': "No active session"
                }
            
            session_id = self.current_session_id
            end_time = datetime.now().isoformat()
            
            # Calculate session duration
            if 'start_time' in self.session_metadata:
                start_dt = datetime.fromisoformat(self.session_metadata['start_time'])
                end_dt = datetime.fromisoformat(end_time)
                duration_seconds = (end_dt - start_dt).total_seconds()
            else:
                duration_seconds = None
            
            # Clear session state
            self.current_session_id = None
            session_metadata = self.session_metadata.copy()
            self.session_metadata = {}
            
            logger.info(f"Session ended: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'end_time': end_time,
                'duration_seconds': duration_seconds,
                'final_metadata': session_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to end session: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.
        
        Returns:
            Session information dictionary
        """
        return {
            'has_active_session': self.current_session_id is not None,
            'session_id': self.current_session_id,
            'metadata': self.session_metadata.copy(),
            'engine_count': len(self.engines),
            'sync_enabled': self.sync_manager.sync_enabled
        }