# API Reference

## Overview

This document provides comprehensive API documentation for all public classes and methods in the PrecipGen library.

## Module Structure

```
precipgen/
├── config/           # Configuration management
├── data/            # Data loading and validation
├── engines/         # Core simulation engines
├── api/            # Standardized API interfaces
└── utils/          # Utilities and exceptions
```

## Configuration Module

### precipgen.config.PrecipGenConfig

Main configuration class for the PrecipGen library.

```python
class PrecipGenConfig:
    """
    Main configuration container for PrecipGen library.
    
    Manages data sources, quality thresholds, and analysis parameters.
    """
```

#### Constructor

```python
def __init__(self, config_dict: Dict = None, config_file: str = None)
```

**Parameters:**
- `config_dict` (dict, optional): Configuration dictionary
- `config_file` (str, optional): Path to configuration file (JSON or YAML)

**Example:**
```python
import precipgen as pg

# From dictionary
config = pg.PrecipGenConfig({
    'data_sources': {
        'site1': {'file_path': 'data.csv'}
    },
    'wet_day_threshold': 0.001
})

# From file
config = pg.PrecipGenConfig(config_file='config.yaml')
```

#### Methods

##### validate()

```python
def validate(self) -> List[str]
```

Validate configuration and return list of errors.

**Returns:**
- `List[str]`: List of validation error messages (empty if valid)

**Example:**
```python
errors = config.validate()
if errors:
    for error in errors:
        print(f"Error: {error}")
```

##### get_data_source()

```python
def get_data_source(self, site_id: str) -> DataSourceConfig
```

Get data source configuration for a specific site.

**Parameters:**
- `site_id` (str): Site identifier

**Returns:**
- `DataSourceConfig`: Data source configuration object

**Raises:**
- `KeyError`: If site_id not found in configuration

##### set_bulk_local_mode()

```python
def set_bulk_local_mode(self, directory: str, station_ids: List[str])
```

Configure bulk local mode for GHCN data processing.

**Parameters:**
- `directory` (str): Directory containing GHCN .dly files
- `station_ids` (List[str]): List of GHCN station identifiers

**Example:**
```python
config.set_bulk_local_mode(
    directory='data/ghcn/',
    station_ids=['USC00123456', 'USC00789012']
)
```

### precipgen.config.DataSourceConfig

Configuration for individual data sources.

```python
class DataSourceConfig:
    """Configuration for a single data source."""
    
    def __init__(self, source_type: str, file_path: str, **kwargs)
```

**Attributes:**
- `source_type` (str): Type of data source ('csv', 'ghcn')
- `file_path` (str): Path to data file
- `column_mapping` (dict): Column name mappings for CSV files
- `station_id` (str): GHCN station identifier

### precipgen.config.QualityConfig

Data quality configuration and thresholds.

```python
class QualityConfig:
    """Configuration for data quality assessment."""
    
    def __init__(self, min_completeness: float = 0.8, **kwargs)
```

**Attributes:**
- `min_completeness` (float): Minimum data completeness threshold (0-1)
- `max_missing_consecutive` (int): Maximum consecutive missing days
- `physical_bounds` (dict): Physical bounds for precipitation values

## Data Module

### precipgen.data.GHCNParser

Parser for GHCN Daily (.dly) format files.

```python
class GHCNParser:
    """Parser for GHCN Daily format precipitation data."""
```

#### Methods

##### parse_dly_file()

```python
def parse_dly_file(self, filepath: str) -> pd.DataFrame
```

Parse a GHCN .dly format file.

**Parameters:**
- `filepath` (str): Path to .dly file

**Returns:**
- `pd.DataFrame`: Parsed data with columns [station_id, date, element, value, flag]

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ParseError`: If file format is invalid

**Example:**
```python
parser = pg.GHCNParser()
data = parser.parse_dly_file('USC00123456.dly')
print(data.head())
```

##### extract_precipitation()

```python
def extract_precipitation(self, data: pd.DataFrame) -> pd.Series
```

Extract precipitation data from parsed GHCN data.

**Parameters:**
- `data` (pd.DataFrame): Parsed GHCN data

**Returns:**
- `pd.Series`: Daily precipitation time series (mm)

##### convert_units()

```python
def convert_units(self, precip_tenths_mm: pd.Series) -> pd.Series
```

Convert precipitation from tenths of mm to mm.

**Parameters:**
- `precip_tenths_mm` (pd.Series): Precipitation in tenths of mm

**Returns:**
- `pd.Series`: Precipitation in mm

##### parse_quality_flags()

```python
def parse_quality_flags(self, data: pd.DataFrame) -> pd.DataFrame
```

Parse and interpret GHCN quality flags.

**Parameters:**
- `data` (pd.DataFrame): Parsed GHCN data

**Returns:**
- `pd.DataFrame`: Quality flag information

### precipgen.data.DataValidator

Data quality validation and assessment.

```python
class DataValidator:
    """Validate precipitation data quality."""
    
    def __init__(self, quality_config: QualityConfig)
```

#### Methods

##### assess_data_quality()

```python
def assess_data_quality(self, data: pd.Series) -> QualityReport
```

Comprehensive data quality assessment.

**Parameters:**
- `data` (pd.Series): Precipitation time series

**Returns:**
- `QualityReport`: Detailed quality assessment

**Example:**
```python
validator = pg.DataValidator(config.quality)
report = validator.assess_data_quality(precip_data)

print(f"Completeness: {report.completeness:.1%}")
print(f"Quality score: {report.quality_score:.3f}")
print(f"Recommendation: {report.recommendation}")
```

##### validate_completeness()

```python
def validate_completeness(self, data: pd.Series) -> ValidationResult
```

Check data completeness against thresholds.

##### validate_physical_bounds()

```python
def validate_physical_bounds(self, data: pd.Series) -> ValidationResult
```

Validate precipitation values are within physical bounds.

## Engines Module

### precipgen.engines.AnalyticalEngine

Statistical analysis and parameter estimation engine.

```python
class AnalyticalEngine:
    """
    Engine for analyzing precipitation data and extracting stochastic parameters.
    
    Implements Richardson & Wright (1984) parameter estimation with modern
    enhancements for trend analysis.
    """
    
    def __init__(self, data: pd.Series, wet_day_threshold: float = 0.001)
```

**Parameters:**
- `data` (pd.Series): Daily precipitation time series
- `wet_day_threshold` (float): Threshold for wet day classification (default: 0.001 inches)

#### Methods

##### calculate_monthly_parameters()

```python
def calculate_monthly_parameters(self) -> Dict[int, MonthlyParams]
```

Calculate monthly transition probabilities and Gamma parameters.

**Returns:**
- `Dict[int, MonthlyParams]`: Monthly parameters for each month (1-12)

**Example:**
```python
engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
monthly_params = engine.calculate_monthly_parameters()

for month, params in monthly_params.items():
    print(f"Month {month}:")
    print(f"  P(W|W): {params.p_ww:.3f}")
    print(f"  P(W|D): {params.p_wd:.3f}")
    print(f"  Alpha: {params.alpha:.3f}")
    print(f"  Beta: {params.beta:.3f}")
```

##### perform_sliding_window_analysis()

```python
def perform_sliding_window_analysis(self, window_years: int = 30) -> WindowAnalysis
```

Perform sliding window analysis to detect parameter evolution.

**Parameters:**
- `window_years` (int): Size of sliding window in years

**Returns:**
- `WindowAnalysis`: Results of sliding window analysis

##### extract_trends()

```python
def extract_trends(self, window_results: WindowAnalysis) -> TrendAnalysis
```

Extract linear trends from sliding window results.

**Parameters:**
- `window_results` (WindowAnalysis): Results from sliding window analysis

**Returns:**
- `TrendAnalysis`: Trend slopes and significance tests

##### generate_parameter_manifest()

```python
def generate_parameter_manifest(self) -> Dict
```

Generate complete parameter manifest for simulation.

**Returns:**
- `Dict`: Complete parameter manifest in JSON-serializable format

**Example:**
```python
manifest = engine.generate_parameter_manifest()

# Save to file
import json
with open('parameters.json', 'w') as f:
    json.dump(manifest, f, indent=2)
```

### precipgen.engines.SimulationEngine

Stochastic precipitation generation engine.

```python
class SimulationEngine:
    """
    Stateful engine for generating synthetic precipitation using WGEN algorithm.
    
    Supports both stationary and non-stationary (trend-based) simulation modes.
    """
    
    def __init__(self, parameters: Dict, trend_mode: bool = False, random_seed: int = None)
```

**Parameters:**
- `parameters` (Dict): Parameter manifest from AnalyticalEngine
- `trend_mode` (bool): Enable non-stationary simulation with parameter drift
- `random_seed` (int, optional): Random seed for reproducibility

#### Methods

##### initialize()

```python
def initialize(self, start_date: datetime, initial_wet_state: bool = False)
```

Initialize simulation state.

**Parameters:**
- `start_date` (datetime): Simulation start date
- `initial_wet_state` (bool): Initial wet/dry state

**Example:**
```python
from datetime import datetime

sim = pg.SimulationEngine(manifest, trend_mode=True, random_seed=42)
sim.initialize(datetime(2025, 1, 1), initial_wet_state=False)
```

##### step()

```python
def step(self) -> float
```

Generate precipitation for one day and advance simulation state.

**Returns:**
- `float`: Daily precipitation amount (mm)

**Example:**
```python
# Generate 365 days of synthetic precipitation
daily_precip = []
for day in range(365):
    precip = sim.step()
    daily_precip.append(precip)

print(f"Annual total: {sum(daily_precip):.1f} mm")
```

##### get_current_state()

```python
def get_current_state(self) -> SimulationState
```

Get current simulation state.

**Returns:**
- `SimulationState`: Current state information

##### reset()

```python
def reset(self, start_date: datetime = None)
```

Reset simulation to initial or specified state.

**Parameters:**
- `start_date` (datetime, optional): New start date (uses original if None)

### precipgen.engines.BootstrapEngine

Historical resampling engine.

```python
class BootstrapEngine:
    """
    Engine for generating precipitation by resampling historical data.
    
    Supports random and sequential sampling modes.
    """
    
    def __init__(self, historical_data: pd.Series, mode: str = 'random')
```

**Parameters:**
- `historical_data` (pd.Series): Historical precipitation time series
- `mode` (str): Sampling mode ('random' or 'sequential')

#### Methods

##### initialize()

```python
def initialize(self, start_date: datetime, random_seed: int = None)
```

Initialize bootstrap sampling.

##### step()

```python
def step(self) -> float
```

Get next precipitation value from historical resampling.

##### get_current_year()

```python
def get_current_year(self) -> int
```

Get the historical year currently being sampled.

**Example:**
```python
bootstrap = pg.BootstrapEngine(historical_data, mode='random')
bootstrap.initialize(datetime(2025, 1, 1), random_seed=42)

# Generate synthetic data
synthetic_data = []
for day in range(365):
    precip = bootstrap.step()
    synthetic_data.append(precip)
    
    if day % 30 == 0:  # Check current year monthly
        current_year = bootstrap.get_current_year()
        print(f"Day {day}: Sampling from year {current_year}")
```

## Data Structures

### MonthlyParams

```python
@dataclass
class MonthlyParams:
    """Monthly precipitation parameters."""
    p_ww: float      # P(wet|wet) transition probability
    p_wd: float      # P(wet|dry) transition probability  
    alpha: float     # Gamma shape parameter
    beta: float      # Gamma scale parameter
```

### SimulationState

```python
@dataclass
class SimulationState:
    """Current state of simulation engine."""
    current_date: datetime    # Current simulation date
    is_wet: bool             # Current wet/dry state
    elapsed_days: int        # Days since simulation start
    current_parameters: MonthlyParams  # Current month parameters
```

### QualityReport

```python
@dataclass
class QualityReport:
    """Data quality assessment report."""
    completeness: float           # Data completeness (0-1)
    quality_score: float         # Overall quality score (0-1)
    out_of_bounds_count: int     # Number of values outside physical bounds
    quality_flags_summary: Dict  # Summary of GHCN quality flags
    recommendation: str          # "ACCEPT" or "REJECT"
```

### TrendAnalysis

```python
@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    seasonal_slopes: Dict[str, Dict[str, float]]    # season -> parameter -> slope
    significance_tests: Dict[str, Dict[str, float]] # season -> parameter -> p_value
    trend_confidence: Dict[str, Dict[str, str]]     # season -> parameter -> significance_level
```

## API Module

### precipgen.api.StandardizedAPI

Standardized interface for external integration.

```python
class StandardizedAPI:
    """
    Standardized API for external model integration.
    
    Provides consistent data exchange formats and synchronization capabilities.
    """
    
    def __init__(self, config: PrecipGenConfig)
```

#### Methods

##### get_simulation_data()

```python
def get_simulation_data(self, engine: SimulationEngine, days: int) -> Dict
```

Get simulation data in standardized format.

**Parameters:**
- `engine` (SimulationEngine): Configured simulation engine
- `days` (int): Number of days to simulate

**Returns:**
- `Dict`: Standardized data format with arrays and metadata

##### synchronize_with_external()

```python
def synchronize_with_external(self, external_date: datetime, engine: SimulationEngine)
```

Synchronize internal simulation with external model clock.

**Example:**
```python
api = pg.StandardizedAPI(config)
sim = pg.SimulationEngine(manifest)

# Get standardized simulation data
data = api.get_simulation_data(sim, days=365)
print(f"Data keys: {list(data.keys())}")
print(f"Precipitation shape: {data['precipitation'].shape}")

# Synchronize with external model
external_date = datetime(2025, 6, 15)
api.synchronize_with_external(external_date, sim)
```

## Utilities Module

### precipgen.utils.exceptions

Custom exception classes for error handling.

```python
class PrecipGenError(Exception):
    """Base exception for PrecipGen library."""

class ConfigurationError(PrecipGenError):
    """Configuration validation errors."""

class DataError(PrecipGenError):
    """Data loading and validation errors."""

class ParseError(DataError):
    """Data parsing errors."""

class ValidationError(DataError):
    """Data validation errors."""

class EstimationError(PrecipGenError):
    """Parameter estimation errors."""

class InsufficientDataError(EstimationError):
    """Insufficient data for reliable estimation."""

class SimulationError(PrecipGenError):
    """Simulation runtime errors."""

class StateError(SimulationError):
    """Invalid simulation state errors."""
```

### precipgen.utils.logging_config

Logging configuration utilities.

```python
def setup_logging(level: str = 'INFO', log_file: str = None)
```

Configure logging for PrecipGen library.

**Parameters:**
- `level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `log_file` (str, optional): Path to log file

**Example:**
```python
import precipgen.utils.logging_config as log_config

# Enable debug logging
log_config.setup_logging(level='DEBUG')

# Log to file
log_config.setup_logging(level='INFO', log_file='precipgen.log')
```

## Usage Patterns

### Basic Analysis Workflow

```python
import precipgen as pg
from datetime import datetime

# 1. Configure
config = pg.PrecipGenConfig({
    'data_sources': {
        'site1': {'file_path': 'precipitation.csv'}
    }
})

# 2. Load and validate data
data = pd.read_csv('precipitation.csv', index_col='date', parse_dates=True)
validator = pg.DataValidator(config.quality)
quality_report = validator.assess_data_quality(data['precipitation'])

# 3. Analyze
engine = pg.AnalyticalEngine(data['precipitation'])
manifest = engine.generate_parameter_manifest()

# 4. Simulate
sim = pg.SimulationEngine(manifest, trend_mode=True, random_seed=42)
sim.initialize(datetime(2025, 1, 1))

# 5. Generate synthetic data
synthetic_data = [sim.step() for _ in range(365)]
```

### GHCN Data Processing

```python
# Parse GHCN data
parser = pg.GHCNParser()
ghcn_data = parser.parse_dly_file('USC00123456.dly')
precip_data = parser.extract_precipitation(ghcn_data)

# Quality assessment
validator = pg.DataValidator(config.quality)
quality_report = validator.assess_data_quality(precip_data)

if quality_report.recommendation == 'ACCEPT':
    # Proceed with analysis
    engine = pg.AnalyticalEngine(precip_data)
    manifest = engine.generate_parameter_manifest()
else:
    print(f"Data quality insufficient: {quality_report.quality_score:.3f}")
```

### Non-Stationary Simulation

```python
# Enable trend analysis
engine = pg.AnalyticalEngine(precip_data)
window_analysis = engine.perform_sliding_window_analysis(window_years=30)
trend_analysis = engine.extract_trends(window_analysis)
manifest = engine.generate_parameter_manifest()

# Non-stationary simulation
sim = pg.SimulationEngine(manifest, trend_mode=True)
sim.initialize(datetime(2025, 1, 1))

# Long-term simulation with parameter drift
long_term_data = []
for year in range(50):
    year_data = [sim.step() for _ in range(365)]
    annual_total = sum(year_data)
    long_term_data.append(annual_total)
    print(f"Year {year+1}: {annual_total:.1f} mm")
```

### Integration with External Models

```python
class HydrologyModel:
    def __init__(self, precip_manifest):
        self.precip_sim = pg.SimulationEngine(precip_manifest)
        self.precip_sim.initialize(datetime(2025, 1, 1))
        self.soil_moisture = 50.0  # Initial soil moisture
    
    def daily_step(self):
        # Get precipitation
        precip = self.precip_sim.step()
        
        # Update hydrology
        infiltration = min(precip, 10.0)  # Max 10mm infiltration
        runoff = max(0, precip - infiltration)
        
        # Update soil moisture
        self.soil_moisture += infiltration
        self.soil_moisture *= 0.95  # Daily evaporation
        
        return {
            'precipitation': precip,
            'runoff': runoff,
            'soil_moisture': self.soil_moisture
        }

# Use integrated model
model = HydrologyModel(manifest)
results = [model.daily_step() for _ in range(365)]
```

## Error Handling

### Exception Hierarchy

All PrecipGen exceptions inherit from `PrecipGenError`:

```python
try:
    config = pg.PrecipGenConfig(invalid_config)
except pg.ConfigurationError as e:
    print(f"Configuration error: {e}")
except pg.PrecipGenError as e:
    print(f"PrecipGen error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Common Error Patterns

```python
# Handle insufficient data
try:
    engine = pg.AnalyticalEngine(sparse_data)
    params = engine.calculate_monthly_parameters()
except pg.InsufficientDataError as e:
    print(f"Not enough data: {e}")
    # Use default parameters or longer time period

# Handle file errors
try:
    parser = pg.GHCNParser()
    data = parser.parse_dly_file('missing_file.dly')
except FileNotFoundError:
    print("Data file not found")
except pg.ParseError as e:
    print(f"File format error: {e}")

# Handle simulation errors
try:
    sim = pg.SimulationEngine(invalid_manifest)
    precip = sim.step()
except pg.StateError as e:
    print(f"Simulation state error: {e}")
    sim.reset()  # Reset to initial state
```

## Performance Considerations

### Memory Optimization

```python
# Use generators for large datasets
def process_large_dataset(file_path):
    chunk_size = 10000
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

# Process in chunks
for chunk in process_large_dataset('large_file.csv'):
    engine = pg.AnalyticalEngine(chunk['precipitation'])
    # Process chunk
```

### Computational Efficiency

```python
# Pre-compute parameters for multiple simulations
manifest = engine.generate_parameter_manifest()

# Run multiple realizations efficiently
realizations = []
for seed in range(100):
    sim = pg.SimulationEngine(manifest, random_seed=seed)
    sim.initialize(datetime(2025, 1, 1))
    realization = [sim.step() for _ in range(365)]
    realizations.append(realization)

# Analyze ensemble statistics
ensemble_mean = np.mean(realizations, axis=0)
ensemble_std = np.std(realizations, axis=0)
```