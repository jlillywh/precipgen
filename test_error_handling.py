#!/usr/bin/env python3
"""
Simple test script to verify comprehensive error handling implementation.
"""

import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Test imports
try:
    from precipgen.config import PrecipGenConfig
    from precipgen.data.validator import DataValidator
    from precipgen.config.quality_config import QualityConfig
    from precipgen.engines.simulation import SimulationEngine
    from precipgen.engines.analytical import AnalyticalEngine, ParameterManifest, MonthlyParams
    from precipgen.utils.exceptions import *
    from precipgen.utils.logging_config import configure_logging, get_logger
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Configure logging
configure_logging(level="INFO")
logger = get_logger('test_error_handling')

def test_configuration_error_handling():
    """Test configuration error handling."""
    print("\n=== Testing Configuration Error Handling ===")
    
    try:
        # This should fail with validation error - use invalid threshold
        config = PrecipGenConfig({'wet_day_threshold': -1.0})
        print("✗ Expected ConfigValidationError but got none")
        return False
    except ConfigValidationError as e:
        print(f"✓ ConfigValidationError caught correctly: {e.message}")
        print(f"  Details: {e.details}")
        print(f"  Guidance: {e.guidance}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_data_validation_error_handling():
    """Test data validation error handling."""
    print("\n=== Testing Data Validation Error Handling ===")
    
    try:
        quality_config = QualityConfig()
        validator = DataValidator(quality_config)
        
        # Test with empty data
        empty_data = pd.Series([], dtype=float)
        result = validator.validate_completeness(empty_data, site_id="TEST001")
        
        if not result.is_valid and "Insufficient data for parameter estimation" in result.errors[0]:
            print("✓ Empty data validation handled correctly")
        else:
            print(f"✗ Unexpected validation result: {result}")
            return False
        
        # Test with invalid data
        invalid_data = pd.Series([1000, -50, np.inf, np.nan, 500])
        result = validator.validate_physical_bounds(invalid_data, site_id="TEST002")
        
        if result.warnings or result.errors:
            print("✓ Invalid data bounds validation handled correctly")
            print(f"  Warnings: {len(result.warnings)}")
            print(f"  Errors: {len(result.errors)}")
        else:
            print("✗ Expected validation warnings/errors but got none")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_error_handling():
    """Test simulation engine error handling."""
    print("\n=== Testing Simulation Error Handling ===")
    
    try:
        # Create minimal parameter manifest
        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'data_period': ['2020-01-01', '2020-12-31'],
            'wet_day_threshold': 0.001
        }
        
        # Create monthly parameters
        monthly_params = {}
        for month in range(1, 13):
            monthly_params[month] = MonthlyParams(
                p_ww=0.5, p_wd=0.3, alpha=1.2, beta=5.0
            )
        
        manifest = ParameterManifest(
            metadata=metadata,
            overall_parameters=monthly_params,
            trend_analysis=None,
            sliding_window_stats=None
        )
        
        # Test simulation engine initialization
        engine = SimulationEngine(manifest, trend_mode=False, random_seed=42)
        print("✓ Simulation engine created successfully")
        
        # Test error when not initialized
        try:
            engine.step()
            print("✗ Expected StateError but got none")
            return False
        except StateError as e:
            print(f"✓ StateError caught correctly: {e.message}")
            print(f"  Recovery possible: {e.details.get('recovery_possible', False)}")
        
        # Test proper initialization and operation
        engine.initialize(datetime(2020, 1, 1), initial_wet_state=False)
        precip = engine.step()
        
        if isinstance(precip, float) and precip >= 0:
            print(f"✓ Simulation step successful: {precip:.3f} mm")
        else:
            print(f"✗ Invalid precipitation value: {precip}")
            return False
        
        # Test error statistics
        stats = engine.get_error_statistics()
        print(f"✓ Error statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_analytical_engine_error_handling():
    """Test analytical engine error handling."""
    print("\n=== Testing Analytical Engine Error Handling ===")
    
    try:
        # Create test data with some issues
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        precip_data = np.random.gamma(1.2, 5.0, len(dates))
        
        # Introduce some data quality issues
        precip_data[50:60] = np.nan  # Missing data
        precip_data[100] = -5.0      # Invalid negative value
        precip_data[200] = 1000.0    # Extreme value
        
        data_series = pd.Series(precip_data, index=dates)
        
        # Create analytical engine
        engine = AnalyticalEngine(data_series, wet_day_threshold=0.001)
        print("✓ Analytical engine created successfully")
        
        # Test analysis with problematic data
        monthly_params = engine.calculate_monthly_parameters()
        print(f"✓ Monthly parameters calculated: {len(monthly_params)} months")
        
        # Test analysis report generation
        report = engine.generate_analysis_report()
        
        if report['analysis_status'] == 'success':
            print("✓ Analysis report generated successfully")
            if report['warnings']:
                print(f"  Warnings detected: {len(report['warnings'])}")
        else:
            print(f"✗ Analysis failed: {report.get('errors', [])}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Analytical engine test failed: {e}")
        traceback.print_exc()
        return False

def test_logging_functionality():
    """Test logging functionality."""
    print("\n=== Testing Logging Functionality ===")
    
    try:
        logger = get_logger('test_component')
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        print("✓ Logging functionality working")
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def main():
    """Run all error handling tests."""
    print("PrecipGen Library - Comprehensive Error Handling Test")
    print("=" * 60)
    
    tests = [
        test_configuration_error_handling,
        test_data_validation_error_handling,
        test_simulation_error_handling,
        test_analytical_engine_error_handling,
        test_logging_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            print(f"✗ {test_func.__name__} crashed: {e}")
            traceback.print_exc()
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All error handling tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())