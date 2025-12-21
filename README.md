# PrecipGen Library

A modular, high-performance Python library for stochastic precipitation generation designed for integration into larger simulation frameworks.

## Overview

PrecipGen implements the Richardson & Wright (1984) WGEN algorithm with modern enhancements including:

- **Non-stationary trend analysis** for climate change modeling
- **GHCN data format support** for standard climate datasets  
- **Flexible sampling modes** (random, sequential bootstrap)
- **Stateful operation** for seamless integration into dynamic simulations

## Architecture

The library is organized around four core pillars:

1. **Data Management**: Configuration, loading, and validation of climate datasets
2. **Historical Resampling**: Bootstrap engine for replaying historical sequences
3. **Parameter Analysis**: Extraction of stochastic parameters with trend detection
4. **Synthetic Generation**: Stateful simulation engine using Markov chains and Gamma distributions

## Installation

```bash
pip install precipgen
```

For development:

```bash
git clone https://github.com/precipgen/precipgen.git
cd precipgen
pip install -e .[dev]
```

## Quick Start

### Option 1: Use Built-in Data Download

```python
import precipgen as pg
from datetime import datetime

# Find and download GHCN data automatically
stations = pg.find_nearby_stations(40.7128, -74.0060, radius_km=50)  # NYC area
file_path = pg.download_station(stations[0]['id'])  # Download closest station

# Parse GHCN data
parser = pg.GHCNParser(file_path)
ghcn_data = parser.parse_dly_file(file_path)
precip_data = parser.extract_precipitation(ghcn_data)

# Analyze and generate synthetic data
engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
manifest = engine.generate_parameter_manifest()

sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))
synthetic_data = [sim.step() for _ in range(365)]

print(f"Generated {len(synthetic_data)} days of synthetic precipitation")
```

### Option 2: Use Your Own Data

```python
import precipgen as pg
from datetime import datetime

# Load and configure data
config = pg.PrecipGenConfig({
    'data_sources': {
        'site1': {'file_path': 'data/precipitation.csv'}
    }
})

# Analyze historical data
engine = pg.AnalyticalEngine(data, wet_day_threshold=0.001)
parameters = engine.generate_parameter_manifest()

# Generate synthetic precipitation
sim = pg.SimulationEngine(parameters, trend_mode=True)
sim.initialize(datetime(2025, 1, 1))

# Generate daily values
for day in range(365):
    precip_mm = sim.step()
    print(f"Day {day+1}: {precip_mm:.2f} mm")
```

## Features

### 🌍 Easy Data Access
- **Built-in GHCN downloader**: Find and download weather station data automatically
- **Location-based search**: Find stations near any latitude/longitude  
- **Name-based search**: Find stations by city or station name
- **Quality validation**: Automatic data quality assessment and recommendations

### 📊 Data Sources
- **GHCN .dly format** with automatic parsing and unit conversion
- **CSV files** with flexible column mapping
- **Bulk local mode** for processing multiple GHCN stations
- **Quality validation** with configurable thresholds

### Analysis Capabilities
- **Monthly parameter estimation** using method of moments
- **Sliding window analysis** for temporal parameter evolution
- **Trend detection** with statistical significance testing
- **Parameter manifest generation** in JSON format

### Simulation Modes
- **Bootstrap resampling** (random or sequential)
- **WGEN synthetic generation** with Markov chains
- **Non-stationary simulation** with parameter drift
- **Stateful operation** for integration with external models

## Documentation

### Quick Links
- **[Installation Guide](docs/installation.md)** - Complete installation instructions
- **[User Guide](docs/user_guide.md)** - Step-by-step tutorials and usage examples
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Mathematical Foundation](docs/mathematical_foundation.md)** - Richardson & Wright algorithms and formulas
- **[Algorithm Documentation](docs/algorithms.md)** - Detailed algorithm descriptions
- **[Examples](docs/examples/)** - Working code examples and tutorials

### Getting Started
1. **New users**: Start with [Getting GHCN Data](docs/getting_ghcn_data.md) and [Tutorial 1: Getting Started](docs/examples/tutorial_01_getting_started.py)
2. **Researchers**: Review [Mathematical Foundation](docs/mathematical_foundation.md)
3. **Developers**: Check [API Reference](docs/api_reference.md) and [Contributing Guide](CONTRIBUTING.md)

Full documentation is also available at [precipgen.readthedocs.io](https://precipgen.readthedocs.io)

## License

MIT License - see LICENSE file for details.

## Citation

If you use PrecipGen in your research, please cite:

```
PrecipGen Development Team. (2024). PrecipGen: A modular stochastic weather generation library. 
Version 0.1.0. https://github.com/precipgen/precipgen
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.