# PrecipGen Library Design Document

## Overview

PrecipGen is a modular, high-performance Python library for stochastic precipitation generation designed for integration into larger simulation frameworks. The library implements the Richardson & Wright (1984) WGEN algorithm with modern enhancements including non-stationary trend analysis, GHCN data format support, and flexible sampling modes.

The library is architected around four core pillars:
1. **Data Management**: Configuration, loading, and validation of climate datasets
2. **Historical Resampling**: Bootstrap engine for replaying historical sequences
3. **Parameter Analysis**: Extraction of stochastic parameters with trend detection
4. **Synthetic Generation**: Stateful simulation engine using Markov chains and Gamma distributions

## Architecture

The library follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Configuration  │  Bootstrap     │  Analysis     │  Simulation │
│  System         │  Engine        │  Engine       │  Engine     │
├─────────────────────────────────────────────────────────────┤
│                    Data Management Layer                    │
│  GHCN Parser    │  CSV Loader    │  Validator    │  Registry   │
├─────────────────────────────────────────────────────────────┤
│                    Mathematical Core Layer                  │
│  Markov Chain   │  Gamma Fitting │  Trend Analysis │  Statistics │
├─────────────────────────────────────────────────────────────┤
│                    Foundation Layer                         │
│  NumPy/SciPy    │  Pandas        │  JSON/YAML    │  Logging    │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Stateful Operation**: Engines maintain internal state for seamless integration
2. **Modular Components**: Each pillar operates independently with well-defined interfaces
3. **Standard Formats**: Native support for GHCN data and standard Python data structures
4. **Publication Quality**: Robust error handling, comprehensive testing, and academic documentation
5. **Performance**: Optimized for long-term simulations with minimal memory footprint

## Components and Interfaces

### 1. Configuration System

**Purpose**: Manage dataset paths, parameters, and operational modes

**Core Classes**:
- `PrecipGenConfig`: Main configuration container
- `DataSourceConfig`: Handles file paths and GHCN station mappings
- `QualityConfig`: Data quality thresholds and validation rules

**Key Methods**:
```python
class PrecipGenConfig:
    def __init__(self, config_dict: Dict = None, config_file: str = None)
    def validate(self) -> List[str]  # Returns validation errors
    def get_data_source(self, site_id: str) -> DataSourceConfig
    def set_bulk_local_mode(self, directory: str, station_ids: List[str])
```

### 2. Data Management Layer

**GHCN Parser**:
```python
class GHCNParser:
    def parse_dly_file(self, filepath: str) -> pd.DataFrame
    def extract_precipitation(self, data: pd.DataFrame) -> pd.Series
    def convert_units(self, precip_tenths_mm: pd.Series) -> pd.Series
    def parse_quality_flags(self, data: pd.DataFrame) -> pd.DataFrame
```

**Data Validator**:
```python
class DataValidator:
    def __init__(self, quality_config: QualityConfig)
    def validate_completeness(self, data: pd.Series) -> ValidationResult
    def validate_physical_bounds(self, data: pd.Series) -> ValidationResult
    def assess_data_quality(self, data: pd.Series) -> QualityReport
```

### 3. Bootstrap Engine

**Purpose**: Historical resampling with random or sequential modes

```python
class BootstrapEngine:
    def __init__(self, historical_data: pd.Series, mode: str = 'random')
    def initialize(self, start_date: datetime, random_seed: int = None)
    def step(self) -> float  # Returns daily precipitation
    def get_current_year(self) -> int
    def reset(self, start_date: datetime = None)
```

**Sampling Modes**:
- `random`: Randomly select historical years with replacement
- `sequential`: Cycle through years in chronological order
- `weighted`: Sample based on user-defined year weights

### 4. Analytical Engine

**Purpose**: Extract stochastic parameters and detect trends

```python
class AnalyticalEngine:
    def __init__(self, data: pd.Series, wet_day_threshold: float = 0.001)
    def calculate_monthly_parameters(self) -> Dict[int, MonthlyParams]
    def perform_sliding_window_analysis(self, window_years: int = 30) -> WindowAnalysis
    def extract_trends(self, window_results: WindowAnalysis) -> TrendAnalysis
    def generate_parameter_manifest(self) -> ParameterManifest
```

**Data Structures**:
```python
@dataclass
class MonthlyParams:
    p_ww: float  # P(wet|wet)
    p_wd: float  # P(wet|dry)
    alpha: float  # Gamma shape parameter
    beta: float   # Gamma scale parameter

@dataclass
class TrendAnalysis:
    seasonal_slopes: Dict[str, Dict[str, float]]  # season -> parameter -> slope
    significance_tests: Dict[str, Dict[str, float]]  # p-values
    trend_confidence: Dict[str, Dict[str, str]]  # significance levels
```

### 5. Simulation Engine

**Purpose**: Generate synthetic precipitation using WGEN algorithm

```python
class SimulationEngine:
    def __init__(self, parameters: ParameterManifest, 
                 trend_mode: bool = False, random_seed: int = None)
    def initialize(self, start_date: datetime, initial_wet_state: bool = False)
    def step(self) -> float  # Returns daily precipitation in mm
    def get_current_state(self) -> SimulationState
    def reset(self, start_date: datetime = None)
```

**State Management**:
```python
@dataclass
class SimulationState:
    current_date: datetime
    is_wet: bool
    random_state: tuple
    elapsed_days: int
    current_parameters: MonthlyParams  # Adjusted for trends if enabled
```

## Data Models

### Parameter Manifest Structure

```json
{
  "metadata": {
    "station_id": "USC00123456",
    "data_period": ["1950-01-01", "2020-12-31"],
    "analysis_date": "2024-01-15",
    "wet_day_threshold": 0.001,
    "data_completeness": 0.95
  },
  "overall_parameters": {
    "monthly": {
      "1": {"p_ww": 0.45, "p_wd": 0.25, "alpha": 1.2, "beta": 8.5},
      "2": {"p_ww": 0.42, "p_wd": 0.28, "alpha": 1.1, "beta": 7.8},
      ...
    }
  },
  "trend_analysis": {
    "seasonal_slopes": {
      "Winter": {"p_ww": 0.001, "p_wd": -0.0005, "alpha": 0.002, "beta": 0.01},
      "Spring": {"p_ww": -0.002, "p_wd": 0.001, "alpha": -0.001, "beta": -0.05},
      ...
    },
    "significance": {
      "Winter": {"p_ww": 0.03, "p_wd": 0.15, "alpha": 0.08, "beta": 0.02},
      ...
    }
  },
  "sliding_window_stats": {
    "volatility": {...},
    "reversion_rates": {...},
    "correlation_matrix": {...}
  }
}
```

### GHCN Data Structure

The library handles GHCN .dly format with the following structure:
- Fixed-width format: 11-char station ID + year + month + element + 31 daily values
- Element codes: PRCP (precipitation), TMAX, TMIN, etc.
- Quality flags: blank (good), D (duplicate), G (gap filled), etc.
- Units: Precipitation in tenths of millimeters

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Before defining the correctness properties, I need to analyze each acceptance criterion for testability:

### Property Reflection

After analyzing all acceptance criteria, I identified several areas where properties can be consolidated:

**Redundancy Analysis:**
- Properties 1.1, 1.2, 1.3 all test configuration validation and can be combined into a comprehensive configuration validation property
- Properties 2.1, 2.2, 2.3, 2.5 all test GHCN data parsing and can be combined into a comprehensive GHCN parsing property
- Properties 4.1, 4.2 both test parameter calculation and can be combined into a comprehensive parameter calculation property
- Properties 5.4, 5.5 both test the core WGEN algorithm and can be combined into a comprehensive WGEN simulation property
- Properties 9.4, 9.5 both test parameter drift and can be combined into a comprehensive trend projection property

**Unique Properties Retained:**
- Bootstrap sampling modes (random vs sequential) provide distinct validation value
- Data quality threshold handling is unique and important
- Trend analysis and significance testing are mathematically distinct
- State preservation and API consistency provide unique validation

Based on this analysis, the following correctness properties provide comprehensive coverage without redundancy:

### Correctness Properties

**Property 1: Configuration Validation Completeness**
*For any* configuration input (file paths, Station_IDs, bulk local mode settings), the Configuration_System should correctly validate the input and either accept valid configurations or provide specific error messages for invalid ones
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

**Property 2: GHCN Data Processing Round Trip**
*For any* valid GHCN .dly file, parsing then reconstructing the data should preserve all precipitation values and quality flags, with automatic unit conversion from tenths of mm to mm
**Validates: Requirements 2.1, 2.2, 2.3, 2.5**

**Property 3: Data Quality Threshold Consistency**
*For any* dataset with varying completeness levels, when missing data exceeds the configured threshold, the system should consistently warn or reject the data period
**Validates: Requirements 2.4**

**Property 4: Bootstrap Sampling Mode Correctness**
*For any* historical dataset, random sampling should produce statistically random sequences while sequential sampling should cycle through years in chronological order with proper wraparound
**Validates: Requirements 3.2, 3.3, 3.5**

**Property 5: Leap Year Transition Handling**
*For any* sequence of years including leap years, the Bootstrap_Engine should maintain temporal continuity when transitioning between leap and non-leap years
**Validates: Requirements 3.4**

**Property 6: Parameter Calculation Mathematical Correctness**
*For any* precipitation time series, calculated transition probabilities P(W|W) and P(W|D) should satisfy 0 ≤ P ≤ 1, and Gamma parameters should be positive with correct statistical relationships to the wet-day precipitation distribution
**Validates: Requirements 4.1, 4.2**

**Property 7: Sliding Window Analysis Consistency**
*For any* time series with sufficient data, sliding window analysis should produce consistent parameter estimates for overlapping periods, with proper handling of insufficient data periods
**Validates: Requirements 4.3, 4.5**

**Property 8: Parameter Manifest Completeness**
*For any* analysis result, the Parameter_Manifest should contain all required components (overall parameters, trend slopes, significance tests) in the correct JSON structure
**Validates: Requirements 4.4, 4.6, 9.3**

**Property 9: WGEN Simulation State Consistency**
*For any* parameter set and initial state, the Simulation_Engine should maintain Markov chain consistency where wet/dry transitions follow the specified probabilities and wet-day amounts follow the appropriate Gamma distribution
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

**Property 10: Trend Projection with Physical Bounds**
*For any* trend slopes and simulation duration, parameter drift should follow the formula Parameter(t) = Parameter_baseline + (Trend_Slope × elapsed_time) while ensuring transition probabilities remain in [0,1] and Gamma parameters remain positive
**Validates: Requirements 5.6, 9.4, 9.5**

**Property 11: Trend Analysis Statistical Validity**
*For any* windowed parameter time series, computed regression slopes should be mathematically correct and significance tests should properly distinguish meaningful trends from noise
**Validates: Requirements 9.1, 9.2**

**Property 12: API Data Type Consistency**
*For any* library interaction, all outputs should use standard Python data types (dictionaries, lists, numpy arrays) and maintain consistent data exchange formats
**Validates: Requirements 6.2, 6.5**

**Property 13: External Simulation Synchronization**
*For any* external simulation clock, the library's internal date tracking should remain synchronized throughout the simulation
**Validates: Requirements 6.3**

## Error Handling

The library implements comprehensive error handling across all components:

### Data Loading Errors
- **FileNotFoundError**: Clear messages with suggested file paths
- **ParseError**: Detailed information about malformed GHCN data
- **ValidationError**: Specific guidance on data quality issues
- **UnitConversionError**: Warnings about unexpected data ranges

### Parameter Estimation Errors
- **InsufficientDataError**: Documentation of periods with inadequate data
- **ConvergenceError**: Fallback strategies for Gamma fitting failures
- **TrendAnalysisError**: Handling of degenerate regression cases

### Simulation Errors
- **StateError**: Recovery from invalid internal states
- **BoundsError**: Graceful handling of parameter drift beyond physical limits
- **SynchronizationError**: Clear messages for date/time mismatches

### Configuration Errors
- **ConfigValidationError**: Specific guidance for invalid settings
- **CompatibilityError**: Version and dependency conflict resolution

## Testing Strategy

The library employs a dual testing approach combining unit tests and property-based tests:

### Unit Testing
Unit tests verify specific examples, edge cases, and integration points:
- GHCN file parsing with known test files
- Parameter calculation with validated datasets
- Bootstrap engine initialization and state management
- Error handling with malformed inputs
- API integration with mock external systems

### Property-Based Testing
Property-based tests verify universal properties across all valid inputs using **Hypothesis** for Python:
- Each property-based test runs a minimum of 100 iterations
- Smart generators constrain inputs to realistic ranges
- Tests focus on mathematical correctness and invariant preservation
- Each test is tagged with the corresponding design property number

**Property Test Configuration**:
```python
# Example property test structure
@given(precipitation_data=precipitation_time_series(),
       wet_threshold=floats(min_value=0.001, max_value=1.0))
@settings(max_examples=100)
def test_parameter_calculation_correctness(precipitation_data, wet_threshold):
    """**Feature: precipgen-library, Property 6: Parameter Calculation Mathematical Correctness**"""
    # Test implementation
```

**Generator Strategies**:
- `precipitation_time_series()`: Generates realistic daily precipitation sequences
- `ghcn_station_ids()`: Generates valid 11-character GHCN station identifiers
- `parameter_sets()`: Generates valid WGEN parameter combinations
- `trend_scenarios()`: Generates realistic trend slope combinations

### Integration Testing
- End-to-end workflows from data loading to simulation
- Cross-component state management validation
- External system integration scenarios
- Performance benchmarks for long-term simulations

### Academic Validation
- Comparison with published WGEN results
- Statistical validation of generated sequences
- Peer review of mathematical implementations
- Reproducibility testing with fixed random seeds