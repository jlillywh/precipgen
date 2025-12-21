# PrecipGen User Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Data Sources](#data-sources)
5. [Analysis Workflows](#analysis-workflows)
6. [Simulation Modes](#simulation-modes)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0

### Install from PyPI

```bash
pip install precipgen
```

### Install from Source

```bash
git clone https://github.com/precipgen/precipgen.git
cd precipgen
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/precipgen/precipgen.git
cd precipgen
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
import precipgen as pg
import pandas as pd
from datetime import datetime

# Load precipitation data
data = pd.read_csv('precipitation.csv', index_col='date', parse_dates=True)

# Create configuration
config = pg.PrecipGenConfig({
    'data_sources': {
        'site1': {'file_path': 'precipitation.csv'}
    },
    'wet_day_threshold': 0.001  # inches
})

# Analyze historical data
engine = pg.AnalyticalEngine(data['precipitation'], wet_day_threshold=0.001)
parameters = engine.generate_parameter_manifest()

# Generate synthetic precipitation
sim = pg.SimulationEngine(parameters, random_seed=42)
sim.initialize(datetime(2025, 1, 1))

# Generate 365 days of synthetic data
synthetic_data = []
for day in range(365):
    precip = sim.step()
    synthetic_data.append(precip)

print(f"Generated {len(synthetic_data)} days of synthetic precipitation")
print(f"Mean daily precipitation: {np.mean(synthetic_data):.3f} inches")
```

### Working with GHCN Data

```python
import precipgen as pg

# Configure for GHCN data
config = pg.PrecipGenConfig({
    'data_sources': {
        'USC00123456': {
            'source_type': 'ghcn',
            'file_path': 'data/USC00123456.dly'
        }
    }
})

# Parse GHCN data
parser = pg.GHCNParser()
data = parser.parse_dly_file('data/USC00123456.dly')
precip_data = parser.extract_precipitation(data)

# Validate data quality
validator = pg.DataValidator(config.quality)
quality_report = validator.assess_data_quality(precip_data)

print(f"Data completeness: {quality_report.completeness:.1%}")
print(f"Quality score: {quality_report.quality_score:.3f}")
```

## Configuration

### Configuration Structure

PrecipGen uses a hierarchical configuration system:

```python
config = pg.PrecipGenConfig({
    'data_sources': {
        'site_id': {
            'source_type': 'csv',  # or 'ghcn'
            'file_path': 'path/to/data.csv',
            'column_mapping': {
                'date': 'date',
                'precipitation': 'precip'
            }
        }
    },
    'quality': {
        'min_completeness': 0.8,
        'max_missing_consecutive': 30,
        'physical_bounds': {
            'min_precip': 0.0,
            'max_precip': 1000.0  # mm
        }
    },
    'analysis': {
        'wet_day_threshold': 0.001,  # inches
        'sliding_window_years': 30,
        'trend_significance_level': 0.05
    }
})
```

### Bulk Local Mode

For processing multiple GHCN stations:

```python
config = pg.PrecipGenConfig()
config.set_bulk_local_mode(
    directory='data/ghcn_stations/',
    station_ids=['USC00123456', 'USC00789012', 'USC00345678']
)

# Process all stations
results = {}
for station_id in config.data_sources.keys():
    data_config = config.get_data_source(station_id)
    parser = pg.GHCNParser()
    data = parser.parse_dly_file(data_config.file_path)
    results[station_id] = parser.extract_precipitation(data)
```

### Configuration Validation

```python
# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

## Data Sources

### CSV Files

#### Standard Format

```csv
date,precipitation
2020-01-01,0.0
2020-01-02,2.5
2020-01-03,0.0
2020-01-04,1.2
```

#### Custom Column Mapping

```python
config = pg.PrecipGenConfig({
    'data_sources': {
        'site1': {
            'source_type': 'csv',
            'file_path': 'custom_data.csv',
            'column_mapping': {
                'date': 'Date',
                'precipitation': 'Daily_Precip_mm'
            }
        }
    }
})
```

### GHCN Daily Format

GHCN .dly files are automatically parsed:

```python
# Single station
parser = pg.GHCNParser()
data = parser.parse_dly_file('USC00123456.dly')

# Extract precipitation with quality flags
precip_data = parser.extract_precipitation(data)
quality_flags = parser.parse_quality_flags(data)

# Unit conversion (automatic)
precip_mm = parser.convert_units(precip_data)  # tenths of mm to mm
```

### Data Quality Assessment

```python
validator = pg.DataValidator(config.quality)

# Comprehensive quality assessment
quality_report = validator.assess_data_quality(precip_data)

print(f"Completeness: {quality_report.completeness:.1%}")
print(f"Out of bounds values: {quality_report.out_of_bounds_count}")
print(f"Quality flags summary: {quality_report.quality_flags_summary}")
print(f"Recommendation: {quality_report.recommendation}")

# Detailed validation
completeness_result = validator.validate_completeness(precip_data)
bounds_result = validator.validate_physical_bounds(precip_data)
```

## Analysis Workflows

### Basic Parameter Estimation

```python
# Create analytical engine
engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)

# Calculate overall monthly parameters
monthly_params = engine.calculate_monthly_parameters()

# Display results
for month, params in monthly_params.items():
    print(f"Month {month}:")
    print(f"  P(W|W): {params.p_ww:.3f}")
    print(f"  P(W|D): {params.p_wd:.3f}")
    print(f"  Alpha: {params.alpha:.3f}")
    print(f"  Beta: {params.beta:.3f}")
```

### Sliding Window Analysis

```python
# Perform sliding window analysis
window_analysis = engine.perform_sliding_window_analysis(window_years=30)

# Extract parameter evolution
for result in window_analysis.results:
    year = result.center_year
    params = result.parameters
    print(f"Year {year}: P(W|W) Jan = {params[1].p_ww:.3f}")
```

### Trend Detection

```python
# Extract trends from sliding window results
trend_analysis = engine.extract_trends(window_analysis)

# Display significant trends
for param_name, trends in trend_analysis.seasonal_slopes.items():
    for season, slope in trends.items():
        p_value = trend_analysis.significance_tests[param_name][season]
        if p_value < 0.05:
            print(f"{param_name} {season}: {slope:.6f}/year (p={p_value:.3f})")
```

### Complete Analysis Pipeline

```python
# Generate comprehensive parameter manifest
manifest = engine.generate_parameter_manifest()

# Save to file
import json
with open('parameters.json', 'w') as f:
    json.dump(manifest, f, indent=2)

# Load from file
with open('parameters.json', 'r') as f:
    loaded_manifest = json.load(f)
```

## Simulation Modes

### Bootstrap Resampling

#### Random Sampling

```python
# Create bootstrap engine with random sampling
bootstrap = pg.BootstrapEngine(
    historical_data=precip_data,
    mode='random',
    random_seed=42
)

# Initialize simulation
bootstrap.initialize(datetime(2025, 1, 1))

# Generate synthetic data
synthetic_data = []
for day in range(365):
    precip = bootstrap.step()
    synthetic_data.append(precip)
    
print(f"Current historical year: {bootstrap.get_current_year()}")
```

#### Sequential Sampling

```python
# Sequential mode cycles through historical years
bootstrap = pg.BootstrapEngine(
    historical_data=precip_data,
    mode='sequential'
)

bootstrap.initialize(datetime(2025, 1, 1))

# Generate multiple years
for year in range(5):
    year_data = []
    for day in range(365):
        precip = bootstrap.step()
        year_data.append(precip)
    print(f"Year {year+1} mean: {np.mean(year_data):.3f} mm")
```

### WGEN Synthetic Generation

#### Stationary Simulation

```python
# Create simulation engine
sim = pg.SimulationEngine(
    parameters=manifest,
    trend_mode=False,
    random_seed=42
)

# Initialize with specific start date and state
sim.initialize(
    start_date=datetime(2025, 1, 1),
    initial_wet_state=False
)

# Generate daily values
daily_precip = []
for day in range(1000):  # Generate 1000 days
    precip = sim.step()
    daily_precip.append(precip)
    
    # Check current state periodically
    if day % 100 == 0:
        state = sim.get_current_state()
        print(f"Day {day}: Date={state.current_date}, Wet={state.is_wet}")
```

#### Non-Stationary Simulation with Trends

```python
# Enable trend mode for non-stationary simulation
sim = pg.SimulationEngine(
    parameters=manifest,
    trend_mode=True,  # Enable parameter drift
    random_seed=42
)

sim.initialize(datetime(2025, 1, 1))

# Generate long-term simulation
long_term_data = []
for year in range(50):  # 50-year simulation
    year_data = []
    for day in range(365):
        precip = sim.step()
        year_data.append(precip)
    
    annual_total = sum(year_data)
    long_term_data.append(annual_total)
    print(f"Year {year+1}: {annual_total:.1f} mm")

# Analyze trends in generated data
import numpy as np
years = np.arange(1, 51)
trend_slope = np.polyfit(years, long_term_data, 1)[0]
print(f"Generated trend: {trend_slope:.2f} mm/year")
```

### State Management

```python
# Save and restore simulation state
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))

# Generate some data
for _ in range(100):
    sim.step()

# Save current state
saved_state = sim.get_current_state()

# Continue simulation
for _ in range(100):
    sim.step()

# Reset to saved state
sim.reset(saved_state.current_date)
# Note: Random state cannot be perfectly restored, use checkpointing for exact reproducibility
```

## Advanced Features

### Custom Parameter Manifests

```python
# Create custom parameter manifest
custom_manifest = {
    'metadata': {
        'station_id': 'CUSTOM001',
        'data_period': ['1990-01-01', '2020-12-31'],
        'wet_day_threshold': 0.001
    },
    'overall_parameters': {
        'monthly': {
            '1': {'p_ww': 0.45, 'p_wd': 0.25, 'alpha': 1.2, 'beta': 8.5},
            '2': {'p_ww': 0.42, 'p_wd': 0.28, 'alpha': 1.1, 'beta': 7.8},
            # ... continue for all 12 months
        }
    },
    'trend_analysis': {
        'seasonal_slopes': {
            'Winter': {'p_ww': 0.001, 'p_wd': -0.0005, 'alpha': 0.002, 'beta': 0.01},
            # ... continue for all seasons
        }
    }
}

# Use custom manifest
sim = pg.SimulationEngine(custom_manifest, trend_mode=True)
```

### Integration with External Models

```python
class ExternalModel:
    def __init__(self):
        self.current_date = datetime(2025, 1, 1)
        self.precip_generator = pg.SimulationEngine(manifest, random_seed=42)
        self.precip_generator.initialize(self.current_date)
    
    def daily_step(self):
        # Generate precipitation for current day
        daily_precip = self.precip_generator.step()
        
        # Use precipitation in model calculations
        runoff = self.calculate_runoff(daily_precip)
        soil_moisture = self.update_soil_moisture(daily_precip)
        
        # Advance model time
        self.current_date += timedelta(days=1)
        
        return {
            'date': self.current_date,
            'precipitation': daily_precip,
            'runoff': runoff,
            'soil_moisture': soil_moisture
        }
    
    def calculate_runoff(self, precip):
        # Simplified runoff calculation
        return max(0, precip - 5.0)  # 5mm infiltration capacity
    
    def update_soil_moisture(self, precip):
        # Simplified soil moisture update
        return min(100, self.soil_moisture + precip * 0.8)

# Run integrated model
model = ExternalModel()
results = []
for day in range(365):
    daily_result = model.daily_step()
    results.append(daily_result)
```

### Batch Processing

```python
def process_multiple_stations(station_configs):
    """Process multiple stations in batch"""
    results = {}
    
    for station_id, config in station_configs.items():
        try:
            # Load and validate data
            if config['source_type'] == 'ghcn':
                parser = pg.GHCNParser()
                data = parser.parse_dly_file(config['file_path'])
                precip_data = parser.extract_precipitation(data)
            else:
                precip_data = pd.read_csv(config['file_path'], 
                                        index_col='date', 
                                        parse_dates=True)['precipitation']
            
            # Analyze data
            engine = pg.AnalyticalEngine(precip_data)
            manifest = engine.generate_parameter_manifest()
            
            # Generate synthetic data
            sim = pg.SimulationEngine(manifest, random_seed=42)
            sim.initialize(datetime(2025, 1, 1))
            
            synthetic_data = [sim.step() for _ in range(365)]
            
            results[station_id] = {
                'manifest': manifest,
                'synthetic_data': synthetic_data,
                'statistics': {
                    'mean_annual': sum(synthetic_data),
                    'wet_days': sum(1 for x in synthetic_data if x > 0.001)
                }
            }
            
        except Exception as e:
            print(f"Error processing station {station_id}: {e}")
            results[station_id] = {'error': str(e)}
    
    return results

# Example usage
stations = {
    'USC00123456': {
        'source_type': 'ghcn',
        'file_path': 'data/USC00123456.dly'
    },
    'USC00789012': {
        'source_type': 'ghcn', 
        'file_path': 'data/USC00789012.dly'
    }
}

batch_results = process_multiple_stations(stations)
```

## Troubleshooting

### Common Issues

#### 1. Data Loading Problems

**Problem**: `FileNotFoundError` when loading data
```python
# Solution: Check file path and permissions
import os
file_path = 'data/precipitation.csv'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
if not os.access(file_path, os.R_OK):
    print(f"File not readable: {file_path}")
```

**Problem**: GHCN parsing errors
```python
# Solution: Validate GHCN file format
try:
    parser = pg.GHCNParser()
    data = parser.parse_dly_file('station.dly')
except pg.ParseError as e:
    print(f"GHCN parsing error: {e}")
    # Check file format, encoding, or corruption
```

#### 2. Parameter Estimation Issues

**Problem**: `InsufficientDataError` during parameter estimation
```python
# Solution: Check data completeness and quality
validator = pg.DataValidator(config.quality)
quality_report = validator.assess_data_quality(precip_data)

if quality_report.completeness < 0.8:
    print("Insufficient data completeness")
    print(f"Available: {quality_report.completeness:.1%}")
    print("Consider:")
    print("- Lowering quality thresholds")
    print("- Using longer time period")
    print("- Gap-filling techniques")
```

**Problem**: Invalid parameter estimates
```python
# Solution: Validate parameters and check data quality
try:
    engine = pg.AnalyticalEngine(precip_data)
    monthly_params = engine.calculate_monthly_parameters()
    
    # Check for reasonable parameter ranges
    for month, params in monthly_params.items():
        if params.alpha <= 0 or params.beta <= 0:
            print(f"Month {month}: Invalid Gamma parameters")
        if not (0 <= params.p_ww <= 1):
            print(f"Month {month}: Invalid P(W|W) = {params.p_ww}")
            
except pg.EstimationError as e:
    print(f"Parameter estimation failed: {e}")
```

#### 3. Simulation Problems

**Problem**: Unrealistic synthetic precipitation
```python
# Solution: Validate parameters and check simulation settings
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))

# Generate test data
test_data = [sim.step() for _ in range(1000)]

# Check statistics
mean_precip = np.mean(test_data)
wet_day_freq = sum(1 for x in test_data if x > 0.001) / len(test_data)

print(f"Mean precipitation: {mean_precip:.3f}")
print(f"Wet day frequency: {wet_day_freq:.3f}")

# Compare with historical statistics
historical_mean = precip_data.mean()
historical_wet_freq = (precip_data > 0.001).mean()

print(f"Historical mean: {historical_mean:.3f}")
print(f"Historical wet frequency: {historical_wet_freq:.3f}")
```

#### 4. Memory and Performance Issues

**Problem**: High memory usage with large datasets
```python
# Solution: Use data chunking and generators
def process_large_dataset(file_path, chunk_size=10000):
    """Process large datasets in chunks"""
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk
        precip_chunk = chunk['precipitation']
        
        # Analyze chunk
        engine = pg.AnalyticalEngine(precip_chunk)
        chunk_params = engine.calculate_monthly_parameters()
        
        yield chunk_params

# Process in chunks
all_params = list(process_large_dataset('large_dataset.csv'))
```

**Problem**: Slow simulation performance
```python
# Solution: Use vectorized operations and pre-computation
# Pre-compute parameters for efficiency
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))

# Generate data in batches
batch_size = 1000
total_days = 10000
all_data = []

for batch_start in range(0, total_days, batch_size):
    batch_data = []
    for _ in range(min(batch_size, total_days - batch_start)):
        batch_data.append(sim.step())
    all_data.extend(batch_data)
    
    # Progress reporting
    progress = (batch_start + batch_size) / total_days
    print(f"Progress: {progress:.1%}")
```

### Debugging Tips

#### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# PrecipGen will now provide detailed logging
engine = pg.AnalyticalEngine(precip_data)
manifest = engine.generate_parameter_manifest()
```

#### Validate Intermediate Results

```python
# Check data at each processing step
print("Raw data shape:", precip_data.shape)
print("Data range:", precip_data.min(), "to", precip_data.max())

# Validate after quality control
clean_data = validator.clean_data(precip_data)
print("Clean data shape:", clean_data.shape)

# Check parameter estimation
monthly_params = engine.calculate_monthly_parameters()
for month, params in monthly_params.items():
    print(f"Month {month}: {params}")
```

#### Test with Known Data

```python
# Create synthetic test data with known properties
np.random.seed(42)
test_data = []

for day in range(1000):
    if np.random.random() < 0.3:  # 30% wet days
        precip = np.random.gamma(1.5, 5.0)  # Known Gamma parameters
    else:
        precip = 0.0
    test_data.append(precip)

test_series = pd.Series(test_data)

# Analyze test data
engine = pg.AnalyticalEngine(test_series)
estimated_params = engine.calculate_monthly_parameters()

# Compare with known parameters
print("Known: Alpha=1.5, Beta=5.0, P(wet)=0.3")
print(f"Estimated: Alpha={estimated_params[1].alpha:.2f}, Beta={estimated_params[1].beta:.2f}")
```

### Getting Help

- **Documentation**: Full API documentation at [precipgen.readthedocs.io](https://precipgen.readthedocs.io)
- **Examples**: Check the `examples/` directory for working code samples
- **Issues**: Report bugs at [github.com/precipgen/precipgen/issues](https://github.com/precipgen/precipgen/issues)
- **Discussions**: Ask questions at [github.com/precipgen/precipgen/discussions](https://github.com/precipgen/precipgen/discussions)