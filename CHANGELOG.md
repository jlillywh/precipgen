# Changelog

All notable changes to PrecipGen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-12-21

### Added
- **Monte Carlo Simulation Framework**: New `PrecipitationSimulator` class for comprehensive precipitation modeling
  - Integrated WGEN, bootstrap, and block bootstrap methods
  - Automatic station finding and data loading
  - Built-in parameter estimation and validation
  - Comprehensive statistical analysis and visualization
- **Water Resources Analysis**: Specialized analyzers for practical applications
  - `DroughtAnalyzer`: Drought duration, severity, and frequency analysis
  - `ExtremeEventAnalyzer`: Return period analysis for infrastructure design
  - `SeasonalAnalyzer`: Seasonal water availability and variability assessment
- **Jupyter Integration**: Streamlined workflow for interactive analysis
  - One-line simulation setup and execution
  - Automatic plot generation and data export
  - Template notebooks for common use cases

### Changed
- **API Enhancement**: Added high-level simulation classes to main namespace
- **Documentation**: Added comprehensive examples for water resources applications

## [0.1.4] - 2024-12-21

### Fixed
- **Quality Assessment Bug**: Fixed critical bug in QualityConfig where empty quality_flags_to_reject list was incorrectly defaulting to ['X', 'W']
  - Permissive quality preset now correctly accepts all quality flags as intended
  - Fixed fallback quality assessment to properly use most lenient settings
  - Data with quality flags now passes permissive quality standards correctly

## [0.1.1] - 2024-12-21

### Fixed
- **API Reference Documentation**: Corrected API reference to match actual implementation
  - Fixed QualityReport structure (completeness_percentage, is_acceptable, recommendations)
  - Added missing ParameterManifest class documentation
  - Corrected method signatures (assess_data_quality with quality_flags and site_id parameters)
  - Updated data structure field names to match implementation
  - Added proper import examples and version compatibility notes

### Added
- **Enhanced Documentation**: 
  - Comprehensive error handling examples with proper exception usage
  - Performance optimization guidance for large datasets
  - Batch processing examples for multiple stations
  - Memory management best practices
  - Version compatibility section

### Changed
- **Documentation Structure**: Improved API reference organization and accuracy
- **Examples**: More realistic usage patterns with proper error checking

## [0.1.0] - 2024-01-15

### Added
- Initial release of PrecipGen library
- Core WGEN algorithm implementation following Richardson & Wright (1984)
- GHCN Daily (.dly) format parser with automatic unit conversion
- Data quality validation and assessment tools
- Bootstrap resampling engine (random and sequential modes)
- Analytical engine for parameter estimation
- Sliding window analysis for temporal parameter evolution
- Trend detection and statistical significance testing
- Non-stationary simulation with parameter drift
- Stateful simulation engine for external model integration
- Comprehensive error handling and logging
- Property-based testing framework using Hypothesis
- Configuration system with validation
- Standardized API for external integration

### Core Features
- **Data Management**: GHCN parser, CSV loader, quality validation
- **Parameter Analysis**: Monthly parameter estimation, trend analysis
- **Simulation Modes**: WGEN synthetic generation, bootstrap resampling
- **Non-Stationary Modeling**: Trend-based parameter drift over time
- **Integration Support**: Standardized APIs, state management

### Mathematical Implementation
- Markov chain wet/dry state transitions
- Gamma distribution parameter estimation (method of moments)
- Linear trend analysis with statistical significance testing
- Physical bounds enforcement for parameter drift
- Seasonal parameter stratification

### Data Format Support
- GHCN Daily (.dly) format with quality flag parsing
- CSV files with flexible column mapping
- Bulk local mode for multiple station processing
- Automatic unit conversion (tenths of mm to mm)

### Quality Assurance
- Comprehensive unit test suite
- Property-based testing for mathematical correctness
- Statistical validation against published results
- Error handling for edge cases and invalid inputs

### Documentation
- Complete API documentation
- Mathematical foundation with Richardson & Wright references
- User guide with examples
- Algorithm documentation with assumptions and limitations

## [0.0.1] - 2023-12-01

### Added
- Initial project structure
- Basic WGEN algorithm skeleton
- Development environment setup
- Testing framework configuration

---

## Release Notes

### Version 0.1.0 - Initial Release

This is the first stable release of PrecipGen, providing a complete implementation of the Richardson & Wright (1984) weather generation methodology with modern enhancements.

**Key Capabilities:**
- Generate synthetic daily precipitation using established meteorological algorithms
- Support for both stationary and non-stationary (climate change) scenarios
- Native GHCN data format support with quality assessment
- Flexible bootstrap resampling for historical data replay
- Comprehensive trend analysis using sliding window techniques
- Stateful operation for seamless integration with external models

**Validation:**
- Tested against published WGEN results
- Property-based testing ensures mathematical correctness
- Comprehensive error handling for production use
- Statistical validation of generated sequences

**Integration:**
- Clean Python APIs with standard data types
- Standardized data exchange formats
- External simulation synchronization capabilities
- Modular architecture for extensibility

This release provides a solid foundation for stochastic weather generation in climate modeling, hydrological studies, and agricultural applications.

---

## Development Milestones

### Phase 1: Core Implementation (Completed)
- ✅ WGEN algorithm implementation
- ✅ GHCN data parser
- ✅ Parameter estimation methods
- ✅ Basic simulation engine

### Phase 2: Advanced Features (Completed)
- ✅ Trend analysis and non-stationary simulation
- ✅ Bootstrap resampling modes
- ✅ Quality validation system
- ✅ Comprehensive error handling

### Phase 3: Integration and Documentation (Completed)
- ✅ Standardized APIs
- ✅ External model integration support
- ✅ Complete documentation system
- ✅ Working examples and tutorials

### Phase 4: Future Enhancements (Planned)
- 🔄 Multi-site spatial correlation modeling
- 🔄 Additional distribution options (Weibull, mixed exponential)
- 🔄 Enhanced visualization tools
- 🔄 Performance optimizations for large-scale applications
- 🔄 Integration with climate model outputs
- 🔄 Web-based interface for non-programmers

---

## Breaking Changes

### Version 0.1.0
- Initial release - no breaking changes from previous versions

### Future Versions
Breaking changes will be clearly documented here with migration guides.

---

## Deprecation Notices

No current deprecations.

---

## Security Updates

No security issues identified in current release.

---

## Performance Improvements

### Version 0.1.0
- Optimized parameter estimation algorithms
- Efficient sliding window analysis implementation
- Memory-efficient data processing for large datasets
- Vectorized operations using NumPy

---

## Bug Fixes

### Version 0.1.0
- Fixed edge cases in Gamma parameter estimation
- Corrected leap year handling in bootstrap sampling
- Resolved numerical stability issues in trend analysis
- Fixed configuration validation edge cases

---

## Contributors

### Version 0.1.0
- PrecipGen Development Team - Initial implementation
- Mathematical validation and testing
- Documentation and examples

---

## Acknowledgments

- Richardson, C.W., and Wright, D.A. (1984) for the foundational WGEN methodology
- NOAA Global Historical Climatology Network for data format specifications
- Open source Python scientific computing community
- Climate modeling and hydrology research communities for requirements and feedback