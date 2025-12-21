"""
Basic functionality tests for PrecipGen library.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from precipgen import PrecipGenConfig, GHCNParser, DataValidator
from precipgen.config import DataSourceConfig, QualityConfig
from precipgen.engines import BootstrapEngine, AnalyticalEngine, SimulationEngine
from precipgen.utils.exceptions import ConfigValidationError


class TestBasicImports:
    """Test that all components can be imported and instantiated."""
    
    def test_config_creation(self):
        """Test basic configuration creation."""
        config = PrecipGenConfig()
        assert config is not None
        assert config.wet_day_threshold == 0.001
        assert config.bulk_local_mode is False
    
    def test_data_source_config(self):
        """Test data source configuration."""
        data_config = DataSourceConfig(
            file_path="test.csv",
            station_id="USC00123456",
            data_type="csv"
        )
        assert data_config.file_path == "test.csv"
        assert data_config.station_id == "USC00123456"
        assert data_config.data_type == "csv"
    
    def test_quality_config(self):
        """Test quality configuration."""
        quality_config = QualityConfig(
            max_missing_percentage=15.0,
            min_years_required=5
        )
        assert quality_config.max_missing_percentage == 15.0
        assert quality_config.min_years_required == 5
    
    def test_engine_instantiation(self):
        """Test that engines can be instantiated."""
        # Create sample data
        dates = pd.date_range('2000-01-01', '2005-12-31', freq='D')
        data = pd.Series(np.random.gamma(1.5, 5.0, len(dates)), index=dates)
        
        # Test BootstrapEngine
        bootstrap = BootstrapEngine(data, mode='random')
        assert bootstrap.mode == 'random'
        assert len(bootstrap.available_years) > 0
        
        # Test AnalyticalEngine
        analytical = AnalyticalEngine(data, wet_day_threshold=0.001)
        assert analytical.wet_day_threshold == 0.001
        assert len(analytical.data) == len(data)


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_valid_configuration(self):
        """Test validation of valid configuration."""
        config_dict = {
            'wet_day_threshold': 0.001,
            'data_sources': {
                'site1': {
                    'file_path': 'README.md',  # Use existing file
                    'data_type': 'csv'
                }
            }
        }
        
        config = PrecipGenConfig(config_dict)
        errors = config.validate()
        
        # Should have no errors for basic valid config
        assert isinstance(errors, list)
    
    def test_invalid_threshold(self):
        """Test validation catches invalid wet day threshold."""
        try:
            config = PrecipGenConfig({'wet_day_threshold': -1.0})
            # If we get here, the constructor didn't raise an exception
            # This shouldn't happen with our current implementation
            assert False, "Expected ConfigValidationError but got none"
        except ConfigValidationError as e:
            # This is expected - the constructor should raise an error
            assert 'wet_day_threshold' in str(e)
            assert len(e.details['validation_errors']) > 0
    
    def test_missing_data_sources(self):
        """Test validation catches missing data sources."""
        config = PrecipGenConfig({})
        errors = config.validate()
        
        assert len(errors) > 0
        assert any('data source' in error.lower() for error in errors)
    
    def test_bulk_local_mode_directory_scanning(self, tmp_path):
        """Test bulk local mode with directory scanning."""
        # Create test .dly files
        (tmp_path / "USC00123456.dly").write_text("test data")
        (tmp_path / "USC00789012.dly").write_text("test data")
        (tmp_path / "invalid.txt").write_text("not a dly file")
        
        config = PrecipGenConfig()
        config.set_bulk_local_mode(str(tmp_path))  # No station_ids provided
        
        # Should automatically discover station IDs
        assert config.bulk_local_mode is True
        assert len(config.station_ids) == 2
        assert "USC00123456" in config.station_ids
        assert "USC00789012" in config.station_ids
        
        # Should be able to get data sources for discovered stations
        data_source = config.get_data_source("USC00123456")
        assert data_source is not None
        assert data_source.station_id == "USC00123456"
        assert data_source.data_type == "ghcn_dly"
    
    def test_bulk_local_mode_explicit_stations(self, tmp_path):
        """Test bulk local mode with explicitly provided station IDs."""
        # Create test .dly files
        (tmp_path / "USC00123456.dly").write_text("test data")
        (tmp_path / "USC00789012.dly").write_text("test data")
        
        config = PrecipGenConfig()
        config.set_bulk_local_mode(str(tmp_path), ["USC00123456"])
        
        # Should use only the explicitly provided station ID
        assert config.bulk_local_mode is True
        assert len(config.station_ids) == 1
        assert "USC00123456" in config.station_ids
        assert "USC00789012" not in config.station_ids
    
    def test_bulk_local_mode_validation_errors(self, tmp_path):
        """Test bulk local mode validation catches errors."""
        config = PrecipGenConfig()
        
        # Test missing directory
        config.set_bulk_local_mode("/nonexistent/directory")
        errors = config.validate()
        assert any("does not exist" in error for error in errors)
        
        # Test directory with no .dly files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config.set_bulk_local_mode(str(empty_dir))
        errors = config.validate()
        assert any("No valid GHCN .dly files found" in error for error in errors)
    
    def test_get_available_stations(self, tmp_path):
        """Test getting all available station IDs."""
        # Create test .dly files
        (tmp_path / "USC00123456.dly").write_text("test data")
        (tmp_path / "USC00789012.dly").write_text("test data")
        
        config = PrecipGenConfig()
        
        # Add explicit data source
        config.data_sources["explicit_station"] = DataSourceConfig(
            file_path="test.csv",
            station_id="explicit_station",
            data_type="csv"
        )
        
        # Add bulk local mode
        config.set_bulk_local_mode(str(tmp_path))
        
        available_stations = config.get_available_stations()
        assert "explicit_station" in available_stations
        assert "USC00123456" in available_stations
        assert "USC00789012" in available_stations
        assert len(available_stations) == 3


class TestDataValidation:
    """Test data validation functionality."""
    
    def test_data_validator_creation(self):
        """Test data validator can be created."""
        quality_config = QualityConfig()
        validator = DataValidator(quality_config)
        
        assert validator.quality_config == quality_config
    
    def test_completeness_validation(self, sample_precipitation_data):
        """Test data completeness validation."""
        quality_config = QualityConfig(max_missing_percentage=10.0)
        validator = DataValidator(quality_config)
        
        result = validator.validate_completeness(sample_precipitation_data)
        
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'metadata')
    
    def test_physical_bounds_validation(self, sample_precipitation_data):
        """Test physical bounds validation."""
        quality_config = QualityConfig(
            physical_bounds_min=0.0,
            physical_bounds_max=1000.0
        )
        validator = DataValidator(quality_config)
        
        result = validator.validate_physical_bounds(sample_precipitation_data)
        
        assert hasattr(result, 'is_valid')
        assert isinstance(result.metadata, dict)


class TestGHCNParser:
    """Test GHCN data parser functionality."""
    
    def test_ghcn_parser_creation(self, tmp_path):
        """Test GHCN parser can be created with valid file."""
        # Create a sample GHCN file
        ghcn_file = tmp_path / "USC00123456.dly"
        ghcn_content = "USC00123456202001PRCP   50 G   25     0    100 G   75     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 \n"
        ghcn_file.write_text(ghcn_content)
        
        parser = GHCNParser(str(ghcn_file))
        assert parser.file_path == ghcn_file
        assert parser.validate_source() is True
    
    def test_ghcn_parser_invalid_file(self):
        """Test GHCN parser handles invalid file paths."""
        with pytest.raises(FileNotFoundError):
            GHCNParser("/nonexistent/file.dly")
    
    def test_ghcn_dly_file_parsing(self, tmp_path):
        """Test parsing of GHCN .dly format files."""
        # Create a sample GHCN file with multiple months
        ghcn_file = tmp_path / "USC00123456.dly"
        ghcn_content = """USC00123456202001PRCP   50 G   25     0    100 G   75     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 
USC00123456202002PRCP    0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 
USC00123456202003PRCP  200 G  150    75     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 """
        ghcn_file.write_text(ghcn_content)
        
        parser = GHCNParser(str(ghcn_file))
        data = parser.parse_dly_file(str(ghcn_file))
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert 'station_id' in data.columns
        assert 'date' in data.columns
        assert 'element' in data.columns
        assert 'value' in data.columns
        assert 'quality_flag' in data.columns
        
        # Check element code extraction
        assert 'PRCP' in data['element'].unique()
        
        # Check station ID
        assert 'USC00123456' in data['station_id'].unique()
    
    def test_precipitation_extraction(self, tmp_path):
        """Test extraction of precipitation data from parsed GHCN data."""
        # Create a sample GHCN file
        ghcn_file = tmp_path / "USC00123456.dly"
        ghcn_content = "USC00123456202001PRCP   50 G   25     0    100 G   75     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 \n"
        ghcn_file.write_text(ghcn_content)
        
        parser = GHCNParser(str(ghcn_file))
        data = parser.parse_dly_file(str(ghcn_file))
        precip = parser.extract_precipitation(data)
        
        # Check precipitation series
        assert isinstance(precip, pd.Series)
        assert len(precip) > 0
        assert precip.index.dtype == 'datetime64[ns]'
        
        # Check unit conversion (should be in mm, not tenths of mm)
        assert precip.iloc[0] == 5.0  # 50 tenths of mm = 5.0 mm
    
    def test_unit_conversion(self, tmp_path):
        """Test automatic unit conversion from tenths of mm to mm."""
        ghcn_file = tmp_path / "test.dly"
        ghcn_file.write_text("USC00123456202001PRCP   50     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 \n")
        
        parser = GHCNParser(str(ghcn_file))
        
        # Test unit conversion directly
        tenths_mm = pd.Series([50, 100, 250, 0])
        mm = parser.convert_units(tenths_mm)
        
        expected = pd.Series([5.0, 10.0, 25.0, 0.0])
        pd.testing.assert_series_equal(mm, expected)
    
    def test_quality_flag_parsing(self, tmp_path):
        """Test parsing and handling of GHCN quality flags."""
        ghcn_file = tmp_path / "test.dly"
        # Properly formatted GHCN line with various quality flags
        ghcn_content = "USC00123456202001PRCP   50 G    25 D     0 X   100 I    75   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   -9999   \n"
        ghcn_file.write_text(ghcn_content)
        
        parser = GHCNParser(str(ghcn_file))
        data = parser.parse_dly_file(str(ghcn_file))
        data_with_flags = parser.parse_quality_flags(data)
        
        # Check quality flag analysis
        assert 'quality_meaning' in data_with_flags.columns
        assert 'is_suspect' in data_with_flags.columns
        
        # Check that we have the expected quality flags
        unique_flags = set(data['quality_flag'].unique())
        assert 'G' in unique_flags  # Gap filled
        assert 'D' in unique_flags  # Duplicate  
        assert 'X' in unique_flags  # Failed bounds check
        assert 'I' in unique_flags  # Internal consistency check failed
        
        # Check specific flag meanings
        flag_meanings = data_with_flags['quality_meaning'].unique()
        assert 'Gap filled' in flag_meanings  # G flag
        assert 'Duplicate' in flag_meanings   # D flag
        
        # Check suspect flag identification
        suspect_data = data_with_flags[data_with_flags['is_suspect']]
        assert len(suspect_data) > 0  # Should have suspect data
        suspect_flags = suspect_data['quality_flag'].unique()
        assert 'X' in suspect_flags  # Failed bounds check should be suspect
        assert 'I' in suspect_flags  # Internal consistency check failed should be suspect
    
    def test_metadata_extraction(self, tmp_path):
        """Test extraction of metadata from GHCN files."""
        ghcn_file = tmp_path / "USC00123456.dly"
        ghcn_content = "USC00123456202001PRCP   50     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 \n"
        ghcn_file.write_text(ghcn_content)
        
        parser = GHCNParser(str(ghcn_file))
        metadata = parser.get_metadata()
        
        assert 'file_path' in metadata
        assert 'file_size' in metadata
        assert 'data_type' in metadata
        assert 'station_id' in metadata
        
        assert metadata['data_type'] == 'ghcn_dly'
        assert metadata['station_id'] == 'USC00123456'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])