"""
Tests for API standardization functionality.

Tests the standardized API interface to ensure all methods return
standard Python data types and provide consistent data exchange formats.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

from precipgen.api.standardized_api import StandardizedAPI
from precipgen.api.data_exchange import DataExchangeFormat, SimulationState, ParameterSet, ValidationReport
from precipgen.api.synchronization import ExternalSimulationSync


class TestStandardizedAPI:
    """Test standardized API interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api = StandardizedAPI()
        self.sample_data = [0.0, 2.5, 0.0, 1.2, 0.0, 3.1, 0.0, 0.8, 0.0, 1.5] * 10
        self.sample_time_series = {
            'values': self.sample_data,
            'dates': [(datetime(2020, 1, 1) + timedelta(days=i)).isoformat() 
                     for i in range(len(self.sample_data))]
        }
        # Create a proper time series with enough data for analysis (3+ years)
        self.analysis_data = {
            'values': [0.0, 2.5, 0.0, 1.2, 0.0, 3.1, 0.0, 0.8, 0.0, 1.5] * 400,  # More data for analysis
            'dates': [(datetime(2020, 1, 1) + timedelta(days=i)).isoformat() 
                     for i in range(4000)]  # 4000 days (~11 years) of data
        }
    
    def test_configuration_management(self):
        """Test configuration management with standard data types."""
        # Get configuration
        config = self.api.get_configuration()
        assert isinstance(config, dict)
        assert all(isinstance(k, str) for k in config.keys())
        
        # Update configuration
        updates = {'wet_day_threshold': 0.002}
        result = self.api.update_configuration(updates)
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert isinstance(result['is_valid'], bool)
    
    def test_bootstrap_engine_creation(self):
        """Test bootstrap engine creation with standard data types."""
        # Create with time series data (this should work)
        result = self.api.create_bootstrap_engine(self.sample_time_series, mode='random')
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'engine_id' in result
        assert isinstance(result['engine_id'], str)
        assert result['engine_type'] == 'bootstrap'
        
        # Create with different mode
        result2 = self.api.create_bootstrap_engine(self.sample_time_series, mode='sequential')
        assert isinstance(result2, dict)
        assert result2['success'] is True
        assert result2['mode'] == 'sequential'
        
        # Test with simple list data (should work with automatic datetime index)
        result3 = self.api.create_bootstrap_engine(self.sample_data, mode='random')
        assert isinstance(result3, dict)
        assert result3['success'] is True
        assert 'engine_id' in result3
    
    def test_analytical_engine_creation(self):
        """Test analytical engine creation with standard data types."""
        result = self.api.create_analytical_engine(self.analysis_data, wet_day_threshold=0.001)
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'engine_id' in result
        assert result['engine_type'] == 'analytical'
        assert result['wet_day_threshold'] == 0.001
    
    def test_engine_management(self):
        """Test engine management operations."""
        # Create an engine
        create_result = self.api.create_bootstrap_engine(self.sample_time_series)
        engine_id = create_result['engine_id']
        
        # Get engine info
        info = self.api.get_engine_info(engine_id)
        assert isinstance(info, dict)
        assert info['success'] is True
        assert info['engine_id'] == engine_id
        
        # List engines
        engines = self.api.list_engines()
        assert isinstance(engines, dict)
        assert 'engines' in engines
        assert engine_id in engines['engines']
        assert engines['total_count'] >= 1
    
    def test_simulation_operations(self):
        """Test simulation operations with standard data types."""
        # Create bootstrap engine
        create_result = self.api.create_bootstrap_engine(self.sample_time_series)
        engine_id = create_result['engine_id']
        
        # Initialize the engine first
        engine = self.api.engines[engine_id]
        engine.initialize(datetime(2020, 1, 1))
        
        # Run single step
        step_result = self.api.run_simulation_step(engine_id)
        assert isinstance(step_result, dict)
        assert step_result['success'] is True
        assert isinstance(step_result['value'], float)
        assert step_result['data_type'] == 'precipitation_mm'
        
        # Run batch simulation
        batch_result = self.api.run_simulation_batch(engine_id, num_steps=5)
        assert isinstance(batch_result, dict)
        assert batch_result['success'] is True
        assert isinstance(batch_result['values'], list)
        assert len(batch_result['values']) == 5
        assert all(isinstance(v, float) for v in batch_result['values'])
    
    def test_data_analysis(self):
        """Test data analysis with standard data types."""
        # Use smaller window for analysis to avoid data length issues
        analysis_config = {'window_years': 2}  # Use 2-year windows instead of default 30
        result = self.api.analyze_data(self.analysis_data, analysis_config)
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'monthly_parameters' in result
        assert 'trend_analysis' in result
        assert 'parameter_manifest' in result
        
        # Verify all nested data uses standard types
        def check_standard_types(obj):
            if isinstance(obj, dict):
                assert all(isinstance(k, str) for k in obj.keys())
                for v in obj.values():
                    check_standard_types(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_standard_types(item)
            else:
                assert isinstance(obj, (str, int, float, bool, type(None)))
        
        check_standard_types(result['monthly_parameters'])
    
    def test_synchronization_management(self):
        """Test synchronization management with standard data types."""
        start_time = datetime(2020, 1, 1)
        
        # Enable synchronization
        sync_result = self.api.enable_external_sync(start_time.isoformat())
        assert isinstance(sync_result, dict)
        assert sync_result['success'] is True
        assert sync_result['sync_enabled'] is True
        
        # Update external time
        new_time = start_time + timedelta(days=1)
        update_result = self.api.update_external_time(new_time.isoformat())
        assert isinstance(update_result, dict)
        assert update_result['success'] is True
        
        # Get sync status
        status = self.api.get_sync_status()
        assert isinstance(status, dict)
        assert 'sync_enabled' in status
    
    def test_session_management(self):
        """Test session management with standard data types."""
        session_id = "test_session_001"
        metadata = {"test_run": True, "version": "1.0"}
        
        # Start session
        start_result = self.api.start_session(session_id, metadata)
        assert isinstance(start_result, dict)
        assert start_result['success'] is True
        assert start_result['session_id'] == session_id
        
        # Get session info
        info = self.api.get_session_info()
        assert isinstance(info, dict)
        assert info['has_active_session'] is True
        assert info['session_id'] == session_id
        
        # End session
        end_result = self.api.end_session()
        assert isinstance(end_result, dict)
        assert end_result['success'] is True
        assert end_result['session_id'] == session_id
    
    def test_error_handling_returns_standard_types(self):
        """Test that error conditions return standard data types."""
        # Try to get info for non-existent engine
        result = self.api.get_engine_info("non_existent_engine")
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result
        assert isinstance(result['error'], str)
        
        # Try to run step on non-existent engine
        step_result = self.api.run_simulation_step("non_existent_engine")
        assert isinstance(step_result, dict)
        assert step_result['success'] is False
        assert isinstance(step_result['error'], str)


class TestDataExchangeFormat:
    """Test data exchange format utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exchange = DataExchangeFormat()
    
    def test_to_standard_dict(self):
        """Test conversion to standard dictionary format."""
        # Test with simple object
        class TestObj:
            def __init__(self):
                self.value = 42
                self.name = "test"
        
        obj = TestObj()
        result = self.exchange.to_standard_dict(obj)
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())
        assert result['value'] == 42
        assert result['name'] == "test"
    
    def test_numpy_array_conversion(self):
        """Test numpy array conversion."""
        # To numpy array
        data = [1.0, 2.0, 3.0]
        array = self.exchange.to_numpy_array(data)
        assert isinstance(array, np.ndarray)
        
        # From numpy array
        back_to_list = self.exchange.from_numpy_array(array)
        assert isinstance(back_to_list, list)
        assert back_to_list == data
    
    def test_time_series_standardization(self):
        """Test time series standardization."""
        # Test with pandas Series
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
        
        result = self.exchange.standardize_time_series(series)
        assert isinstance(result, dict)
        assert 'values' in result
        assert 'dates' in result
        assert 'has_dates' in result
        assert result['has_dates'] is True
        assert isinstance(result['values'], list)
        assert len(result['values']) == 5


class TestValidationReport:
    """Test validation report functionality."""
    
    def test_validation_report_creation(self):
        """Test validation report creation and usage."""
        report = ValidationReport()
        
        # Add different types of messages
        report.add_error("Test error")
        report.add_warning("Test warning")
        report.add_info("Test info")
        report.set_metadata("test_key", "test_value")
        
        # Convert to dictionary
        result = report.to_dict()
        assert isinstance(result, dict)
        assert result['is_valid'] is False  # Has errors
        assert len(result['errors']) == 1
        assert len(result['warnings']) == 1
        assert len(result['info']) == 1
        assert result['metadata']['test_key'] == "test_value"


class TestExternalSimulationSync:
    """Test external simulation synchronization."""
    
    def test_synchronization_lifecycle(self):
        """Test complete synchronization lifecycle."""
        sync = ExternalSimulationSync()
        start_time = datetime(2020, 1, 1)
        
        # Enable synchronization
        sync.enable_synchronization(start_time)
        assert sync.sync_enabled is True
        
        # Test small time advancement (within tolerance)
        new_time = start_time + timedelta(seconds=30)  # Small advancement within 60s tolerance
        result = sync.update_external_clock(new_time)
        assert result is True  # Should be within tolerance
        
        # Get sync state
        state = sync.get_sync_state()
        assert isinstance(state, dict)
        assert state['sync_enabled'] is True
        
        # Validate synchronization
        validation = sync.validate_synchronization()
        assert isinstance(validation, dict)
        assert 'is_valid' in validation
    
    def test_sync_state_returns_standard_types(self):
        """Test that sync state uses only standard data types."""
        sync = ExternalSimulationSync()
        sync.enable_synchronization(datetime(2020, 1, 1))
        
        state = sync.get_sync_state()
        
        def check_standard_types(obj):
            if isinstance(obj, dict):
                assert all(isinstance(k, str) for k in obj.keys())
                for v in obj.values():
                    check_standard_types(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_standard_types(item)
            else:
                assert isinstance(obj, (str, int, float, bool, type(None)))
        
        check_standard_types(state)


if __name__ == "__main__":
    pytest.main([__file__])