"""
Integration tests for AnalyticalEngine JSON export and error handling.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from datetime import datetime

from precipgen.engines import AnalyticalEngine
from precipgen.engines.analytical import ParameterManifest


class TestParameterManifestSerialization:
    """Test JSON serialization of parameter manifest."""
    
    def test_parameter_manifest_to_dict(self):
        """Test conversion of parameter manifest to dictionary."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Generate manifest
        manifest = engine.generate_parameter_manifest()
        
        # Convert to dictionary
        manifest_dict = manifest.to_dict()
        
        # Verify structure
        assert 'metadata' in manifest_dict
        assert 'overall_parameters' in manifest_dict
        assert 'trend_analysis' in manifest_dict
        assert 'sliding_window_stats' in manifest_dict
        
        # Verify metadata
        assert 'analysis_date' in manifest_dict['metadata']
        assert 'data_period' in manifest_dict['metadata']
        assert 'wet_day_threshold' in manifest_dict['metadata']
        assert 'data_completeness' in manifest_dict['metadata']
        
        # Verify overall parameters
        assert len(manifest_dict['overall_parameters']) == 12
        for month_str, params in manifest_dict['overall_parameters'].items():
            assert 'p_ww' in params
            assert 'p_wd' in params
            assert 'alpha' in params
            assert 'beta' in params
    
    def test_parameter_manifest_to_json(self):
        """Test conversion of parameter manifest to JSON string."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Generate manifest
        manifest = engine.generate_parameter_manifest()
        
        # Convert to JSON
        json_str = manifest.to_json()
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert 'metadata' in parsed
        assert 'overall_parameters' in parsed
    
    def test_parameter_manifest_save_to_file(self):
        """Test saving parameter manifest to JSON file."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Generate manifest
        manifest = engine.generate_parameter_manifest()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            manifest.save_to_file(temp_path)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded = json.load(f)
            
            assert isinstance(loaded, dict)
            assert 'metadata' in loaded
            assert 'overall_parameters' in loaded
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_parameter_manifest_with_trend_analysis(self):
        """Test JSON serialization with trend analysis included."""
        # Create test data with trend
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(42)
        
        # Create trend
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        trend_wet_prob = 0.2 + 0.15 * year_progress
        
        wet_days = np.random.random(len(dates)) < trend_wet_prob
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform comprehensive analysis
        engine.perform_comprehensive_trend_analysis(window_years=3)
        
        # Generate manifest
        manifest = engine.generate_parameter_manifest()
        
        # Convert to dictionary
        manifest_dict = manifest.to_dict()
        
        # Verify trend analysis is included
        assert manifest_dict['trend_analysis'] is not None
        assert 'seasonal_slopes' in manifest_dict['trend_analysis']
        assert 'significance_tests' in manifest_dict['trend_analysis']
        assert 'trend_confidence' in manifest_dict['trend_analysis']
        assert 'regression_type' in manifest_dict['trend_analysis']
        
        # Verify sliding window stats include trend info
        assert manifest_dict['sliding_window_stats'] is not None
        assert 'regression_type' in manifest_dict['sliding_window_stats']
        assert 'trend_validation_performed' in manifest_dict['sliding_window_stats']


class TestAnalysisReporting:
    """Test analysis reporting and error handling."""
    
    def test_generate_analysis_report_success(self):
        """Test generation of analysis report with successful analysis."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate parameters
        engine.calculate_monthly_parameters()
        
        # Generate report
        report = engine.generate_analysis_report()
        
        # Verify report structure
        assert 'analysis_status' in report
        assert 'errors' in report
        assert 'warnings' in report
        assert 'data_quality' in report
        assert 'analysis_results' in report
        
        # Verify success status
        assert report['analysis_status'] == 'success'
        assert len(report['errors']) == 0
        
        # Verify data quality metrics
        assert 'total_data_points' in report['data_quality']
        assert 'data_completeness' in report['data_quality']
        assert 'data_period' in report['data_quality']
        assert 'wet_day_fraction' in report['data_quality']
        
        # Verify analysis results
        assert report['analysis_results']['overall_parameters_calculated'] is True
        assert report['analysis_results']['monthly_parameter_count'] == 12
    
    def test_generate_analysis_report_with_warnings(self):
        """Test analysis report with data quality warnings."""
        # Create sparse data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        # Create very sparse wet days
        wet_days = np.random.random(len(dates)) < 0.02  # Only 2% wet days
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Calculate parameters
        engine.calculate_monthly_parameters()
        
        # Generate report
        report = engine.generate_analysis_report()
        
        # Should have warning about low wet day fraction
        assert len(report['warnings']) > 0
        assert any('wet day fraction' in w.lower() for w in report['warnings'])
    
    def test_generate_analysis_report_empty_data(self):
        """Test analysis report with empty data."""
        # Create empty data
        empty_data = pd.Series([], dtype=float)
        
        engine = AnalyticalEngine(empty_data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Generate report
        report = engine.generate_analysis_report()
        
        # Should have failed status
        assert report['analysis_status'] == 'failed'
        assert len(report['errors']) > 0
        assert any('no data' in e.lower() for e in report['errors'])
    
    def test_generate_analysis_report_with_trend_analysis(self):
        """Test analysis report including trend analysis results."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform comprehensive analysis
        engine.perform_comprehensive_trend_analysis(window_years=3)
        
        # Generate report
        report = engine.generate_analysis_report()
        
        # Verify trend analysis is reported
        assert 'trend_analysis' in report['analysis_results']
        assert report['analysis_results']['trend_analysis']['completed'] is True
        assert 'regression_type' in report['analysis_results']['trend_analysis']
        assert 'seasons_analyzed' in report['analysis_results']['trend_analysis']
        assert 'validation_performed' in report['analysis_results']['trend_analysis']


class TestExportResults:
    """Test export functionality."""
    
    def test_export_results_basic(self):
        """Test basic export of results to files."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        engine.calculate_monthly_parameters()
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = engine.export_results(temp_dir)
            
            # Verify files were created
            assert 'parameter_manifest' in output_files
            assert 'analysis_report' in output_files
            
            assert os.path.exists(output_files['parameter_manifest'])
            assert os.path.exists(output_files['analysis_report'])
            
            # Verify files contain valid JSON
            with open(output_files['parameter_manifest'], 'r') as f:
                manifest_data = json.load(f)
            assert 'metadata' in manifest_data
            assert 'overall_parameters' in manifest_data
            
            with open(output_files['analysis_report'], 'r') as f:
                report_data = json.load(f)
            assert 'analysis_status' in report_data
            assert 'data_quality' in report_data
    
    def test_export_results_with_trend_analysis(self):
        """Test export with complete trend analysis."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2007-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform comprehensive analysis
        engine.perform_comprehensive_trend_analysis(window_years=3)
        
        # Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = engine.export_results(temp_dir)
            
            # Load and verify manifest includes trend analysis
            with open(output_files['parameter_manifest'], 'r') as f:
                manifest_data = json.load(f)
            
            assert manifest_data['trend_analysis'] is not None
            assert 'seasonal_slopes' in manifest_data['trend_analysis']
            
            # Load and verify report includes trend analysis
            with open(output_files['analysis_report'], 'r') as f:
                report_data = json.load(f)
            
            assert report_data['analysis_results']['trend_analysis']['completed'] is True
    
    def test_export_results_selective(self):
        """Test selective export of results."""
        # Create test data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        np.random.seed(42)
        
        wet_days = np.random.random(len(dates)) < 0.3
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        engine.calculate_monthly_parameters()
        
        # Export only manifest
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = engine.export_results(temp_dir, include_json=True, include_report=False)
            
            assert 'parameter_manifest' in output_files
            assert 'analysis_report' not in output_files
            assert os.path.exists(output_files['parameter_manifest'])
        
        # Export only report
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = engine.export_results(temp_dir, include_json=False, include_report=True)
            
            assert 'parameter_manifest' not in output_files
            assert 'analysis_report' in output_files
            assert os.path.exists(output_files['analysis_report'])


class TestComprehensiveWorkflow:
    """Test complete analytical workflow integration."""
    
    def test_complete_analysis_workflow(self):
        """Test complete workflow from data to exported results."""
        # Create realistic test data
        dates = pd.date_range('2000-01-01', '2009-12-31', freq='D')
        np.random.seed(42)
        
        # Generate seasonal pattern with trend
        day_of_year = dates.dayofyear.values
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        years = dates.year.values
        year_progress = (years - years.min()) / (years.max() - years.min())
        trend_factor = 1 + 0.2 * year_progress
        
        wet_prob = 0.25 * seasonal_factor * trend_factor / (seasonal_factor.mean() * 1.1)
        wet_days = np.random.random(len(dates)) < wet_prob
        
        precip_amounts = np.where(
            wet_days,
            np.random.gamma(1.5, 5.0, len(dates)),
            0.0
        )
        
        data = pd.Series(precip_amounts, index=dates)
        
        # Initialize engine
        engine = AnalyticalEngine(data, wet_day_threshold=0.001)
        engine.initialize()
        
        # Perform complete analysis
        engine.calculate_monthly_parameters()
        engine.perform_comprehensive_trend_analysis(window_years=3, regression_type='linear')
        
        # Generate manifest
        manifest = engine.generate_parameter_manifest()
        
        # Verify manifest completeness
        assert manifest.metadata is not None
        assert manifest.overall_parameters is not None
        assert manifest.trend_analysis is not None
        assert manifest.sliding_window_stats is not None
        
        # Generate report
        report = engine.generate_analysis_report()
        
        # Verify report shows success
        assert report['analysis_status'] == 'success'
        assert report['analysis_results']['overall_parameters_calculated'] is True
        assert report['analysis_results']['sliding_window_analysis']['completed'] is True
        assert report['analysis_results']['trend_analysis']['completed'] is True
        
        # Export results
        with tempfile.TemporaryDirectory() as temp_dir:
            output_files = engine.export_results(temp_dir)
            
            # Verify all files created
            assert len(output_files) == 2
            for filepath in output_files.values():
                assert os.path.exists(filepath)
                
                # Verify valid JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)
                assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
