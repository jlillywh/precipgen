#!/usr/bin/env python3
"""
Installation verification script for PrecipGen.

Run this script to verify that PrecipGen is properly installed and working.
"""

import sys
import traceback

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        import precipgen as pg
        print(f"✓ PrecipGen version {pg.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import precipgen: {e}")
        return False
    
    # Test core components
    components = [
        ('PrecipGenConfig', pg.PrecipGenConfig),
        ('AnalyticalEngine', pg.AnalyticalEngine),
        ('SimulationEngine', pg.SimulationEngine),
        ('BootstrapEngine', pg.BootstrapEngine),
        ('DataValidator', pg.DataValidator),
        ('GHCNParser', pg.GHCNParser),
    ]
    
    for name, component in components:
        try:
            # Just check that the class exists
            assert hasattr(pg, name), f"{name} not found in precipgen module"
            print(f"✓ {name} available")
        except Exception as e:
            print(f"✗ {name} not available: {e}")
            return False
    
    return True

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', 'scipy'),
        ('matplotlib', 'plt'),
        ('yaml', 'yaml'),
    ]
    
    for dep_name, import_name in dependencies:
        try:
            if import_name == 'plt':
                import matplotlib.pyplot as plt
            elif import_name == 'yaml':
                import yaml
            else:
                exec(f"import {dep_name} as {import_name}")
            print(f"✓ {dep_name} available")
        except ImportError as e:
            print(f"✗ {dep_name} not available: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic PrecipGen functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import precipgen as pg
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        # Test configuration
        config = pg.PrecipGenConfig()
        print("✓ Configuration creation works")
        
        # Generate test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        precip = np.random.gamma(1.5, 3.0, len(dates)) * (np.random.random(len(dates)) < 0.3)
        data = pd.Series(precip, index=dates)
        print("✓ Test data generation works")
        
        # Test analytical engine
        engine = pg.AnalyticalEngine(data, wet_day_threshold=0.001)
        manifest = engine.generate_parameter_manifest()
        print("✓ Parameter estimation works")
        
        # Test simulation engine
        sim = pg.SimulationEngine(manifest, random_seed=42)
        sim.initialize(datetime(2025, 1, 1))
        synthetic = [sim.step() for _ in range(10)]  # Just 10 days for testing
        print("✓ Simulation engine works")
        
        # Test bootstrap engine
        bootstrap = pg.BootstrapEngine(data, mode='random')
        bootstrap.initialize(datetime(2025, 1, 1), random_seed=42)
        bootstrap_data = [bootstrap.step() for _ in range(10)]
        print("✓ Bootstrap engine works")
        
        print(f"✓ Generated {len(synthetic)} synthetic values and {len(bootstrap_data)} bootstrap values")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all installation tests."""
    print("PrecipGen Installation Verification")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Dependencies Test", test_dependencies),
        ("Functionality Test", test_basic_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("Test Summary:")
    print("-" * 20)
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! PrecipGen is properly installed and working.")
        print("\nNext steps:")
        print("1. Try the examples in docs/examples/")
        print("2. Read the user guide: docs/user_guide.md")
        print("3. Check the API reference: docs/api_reference.md")
    else:
        print("❌ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Ensure you're using the correct Python environment")
        print("2. Try reinstalling: pip install git+https://github.com/jlillywh/precipgen.git")
        print("3. Check the installation guide: docs/installation.md")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())