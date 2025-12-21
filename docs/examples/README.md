# PrecipGen Examples

This directory contains working code examples demonstrating various features of the PrecipGen library.

## Example Files

### Basic Usage
- **[basic_analysis.py](basic_analysis.py)** - Basic parameter estimation and simulation
- **[ghcn_processing.py](ghcn_processing.py)** - Working with GHCN data files
- **[data_validation.py](data_validation.py)** - Data quality assessment and validation

### Advanced Features
- **[trend_analysis.py](trend_analysis.py)** - Non-stationary trend analysis and simulation
- **[bootstrap_sampling.py](bootstrap_sampling.py)** - Historical resampling techniques
- **[batch_processing.py](batch_processing.py)** - Processing multiple stations

### Integration Examples
- **[external_model_integration.py](external_model_integration.py)** - Integration with external simulation models
- **[statistical_validation.py](statistical_validation.py)** - Validating synthetic precipitation against historical data

### Tutorials
- **[tutorial_01_getting_started.py](tutorial_01_getting_started.py)** - Complete beginner tutorial
- **[tutorial_02_advanced_analysis.py](tutorial_02_advanced_analysis.py)** - Advanced analysis techniques
- **[tutorial_03_custom_workflows.py](tutorial_03_custom_workflows.py)** - Building custom analysis workflows

## Running Examples

All examples are self-contained and include sample data generation where needed:

```bash
# Run basic analysis example
python basic_analysis.py

# Run GHCN processing example
python ghcn_processing.py

# Run trend analysis example
python trend_analysis.py
```

## Data Requirements

Most examples generate synthetic test data, but some require external data files:

- **GHCN examples**: Download sample .dly files from NOAA GHCN database
- **CSV examples**: Use provided sample CSV format

## Example Data Format

### CSV Format
```csv
date,precipitation
2020-01-01,0.0
2020-01-02,2.5
2020-01-03,0.0
2020-01-04,1.2
```

### GHCN Station IDs
Common test stations:
- USC00050848 (Boulder, CO)
- USC00305426 (Miami, FL) 
- USC00200032 (Fairbanks, AK)