# Installation Guide

## System Requirements

### Python Version
- Python 3.8 or higher
- Tested on Python 3.8, 3.9, 3.10, 3.11, and 3.12

### Operating Systems
- Windows 10/11
- macOS 10.15 or later
- Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)

### Hardware Requirements
- Minimum: 4 GB RAM, 1 GB disk space
- Recommended: 8 GB RAM, 2 GB disk space
- For large datasets (>50 years): 16 GB RAM recommended

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The easiest way to install PrecipGen is using pip:

```bash
pip install precipgen
```

To install with all optional dependencies:

```bash
pip install precipgen[all]
```

### Method 2: Install from Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/precipgen/precipgen.git
cd precipgen

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### Method 3: Using Conda

If you prefer conda package management:

```bash
# Add conda-forge channel (if not already added)
conda config --add channels conda-forge

# Install PrecipGen
conda install precipgen
```

## Dependencies

### Required Dependencies

PrecipGen requires the following packages:

```
numpy >= 1.19.0
pandas >= 1.3.0
scipy >= 1.7.0
```

These will be automatically installed when you install PrecipGen.

### Optional Dependencies

For enhanced functionality, you can install optional dependencies:

```bash
# For plotting and visualization
pip install matplotlib seaborn

# For advanced statistical analysis
pip install statsmodels

# For faster numerical computations
pip install numba

# For reading Excel files
pip install openpyxl

# For HDF5 file support
pip install tables

# Install all optional dependencies
pip install precipgen[all]
```

## Virtual Environment Setup

### Using venv (Recommended)

Create an isolated environment for PrecipGen:

```bash
# Create virtual environment
python -m venv precipgen_env

# Activate virtual environment
# On Windows:
precipgen_env\Scripts\activate
# On macOS/Linux:
source precipgen_env/bin/activate

# Install PrecipGen
pip install precipgen

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n precipgen python=3.10

# Activate environment
conda activate precipgen

# Install PrecipGen
pip install precipgen

# Deactivate when done
conda deactivate
```

## Verification

### Quick Test

Verify your installation by running:

```python
import precipgen as pg
print(f"PrecipGen version: {pg.__version__}")

# Quick functionality test
config = pg.PrecipGenConfig()
print("✓ PrecipGen installed successfully!")
```

### Run Test Suite

If you installed from source, run the test suite:

```bash
# Install test dependencies
pip install pytest hypothesis

# Run tests
pytest tests/

# Run with coverage
pip install pytest-cov
pytest --cov=precipgen tests/
```

### Example Workflow Test

Test a complete workflow:

```python
import precipgen as pg
import numpy as np
import pandas as pd
from datetime import datetime

# Generate test data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
precip = np.random.gamma(1.5, 3.0, len(dates)) * (np.random.random(len(dates)) < 0.3)
data = pd.Series(precip, index=dates)

# Analyze data
engine = pg.AnalyticalEngine(data)
manifest = engine.generate_parameter_manifest()

# Generate synthetic data
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))
synthetic = [sim.step() for _ in range(365)]

print(f"✓ Generated {len(synthetic)} days of synthetic precipitation")
print(f"Mean: {np.mean(synthetic):.2f} mm/day")
```

## Troubleshooting

### Common Installation Issues

#### Issue: "No module named 'precipgen'"

**Solution:**
```bash
# Ensure you're in the correct environment
pip list | grep precipgen

# If not found, reinstall
pip install --upgrade precipgen
```

#### Issue: Import errors with dependencies

**Solution:**
```bash
# Update all dependencies
pip install --upgrade numpy pandas scipy

# Or reinstall PrecipGen
pip uninstall precipgen
pip install precipgen
```

#### Issue: Permission errors on Windows

**Solution:**
```bash
# Install for current user only
pip install --user precipgen

# Or run as administrator
# Right-click Command Prompt -> "Run as administrator"
pip install precipgen
```

#### Issue: SSL certificate errors

**Solution:**
```bash
# Use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org precipgen

# Or upgrade pip and certificates
pip install --upgrade pip certifi
```

### Platform-Specific Issues

#### Windows

**Issue: Microsoft Visual C++ compiler errors**

**Solution:**
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use pre-compiled wheels
pip install --only-binary=all precipgen
```

#### macOS

**Issue: Xcode command line tools missing**

**Solution:**
```bash
# Install Xcode command line tools
xcode-select --install

# Then install PrecipGen
pip install precipgen
```

#### Linux

**Issue: Missing system libraries**

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++

# Then install PrecipGen
pip install precipgen
```

### Performance Issues

#### Issue: Slow parameter estimation

**Solutions:**
- Install numba for faster computations: `pip install numba`
- Use smaller sliding window sizes for trend analysis
- Process data in chunks for very large datasets

#### Issue: Memory errors with large datasets

**Solutions:**
- Increase system RAM or use a machine with more memory
- Process data in smaller chunks
- Use data compression: `pip install tables` for HDF5 support

### Getting Help

If you encounter issues not covered here:

1. **Check the documentation**: [precipgen.readthedocs.io](https://precipgen.readthedocs.io)
2. **Search existing issues**: [GitHub Issues](https://github.com/precipgen/precipgen/issues)
3. **Create a new issue**: Include your system info, Python version, and error messages
4. **Ask questions**: [GitHub Discussions](https://github.com/precipgen/precipgen/discussions)

## Development Installation

### For Contributors

If you plan to contribute to PrecipGen:

```bash
# Clone repository
git clone https://github.com/precipgen/precipgen.git
cd precipgen

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Development Dependencies

The development installation includes:

```
pytest >= 6.0
hypothesis >= 6.0
black >= 22.0
flake8 >= 4.0
isort >= 5.0
mypy >= 0.900
pre-commit >= 2.0
sphinx >= 4.0
sphinx-rtd-theme >= 1.0
```

## Docker Installation

### Using Docker

For containerized deployment:

```bash
# Pull the official image
docker pull precipgen/precipgen:latest

# Or build from source
git clone https://github.com/precipgen/precipgen.git
cd precipgen
docker build -t precipgen .

# Run container
docker run -it precipgen python -c "import precipgen; print('Success!')"
```

### Docker Compose

For development with Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  precipgen:
    build: .
    volumes:
      - .:/app
      - ./data:/app/data
    working_dir: /app
    command: python -m pytest
```

## Jupyter Notebook Setup

### Installation

```bash
# Install Jupyter
pip install jupyter

# Install PrecipGen
pip install precipgen

# Start Jupyter
jupyter notebook
```

### Example Notebook

Create a new notebook and test:

```python
# Cell 1: Import and test
import precipgen as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PrecipGen version: {pg.__version__}")

# Cell 2: Quick example
# Generate test data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
precip = np.random.gamma(1.5, 3.0, len(dates)) * (np.random.random(len(dates)) < 0.3)
data = pd.Series(precip, index=dates)

# Cell 3: Analysis
engine = pg.AnalyticalEngine(data)
manifest = engine.generate_parameter_manifest()
print("✓ Parameter estimation complete")

# Cell 4: Simulation
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(pd.Timestamp('2025-01-01'))
synthetic = [sim.step() for _ in range(365)]

# Cell 5: Visualization
plt.figure(figsize=(12, 4))
plt.plot(data.iloc[:365], alpha=0.7, label='Historical')
plt.plot(synthetic, alpha=0.7, label='Synthetic')
plt.legend()
plt.title('Historical vs Synthetic Precipitation')
plt.ylabel('Precipitation (mm)')
plt.show()
```

## Next Steps

After successful installation:

1. **Read the User Guide**: `docs/user_guide.md`
2. **Try the tutorials**: `docs/examples/tutorial_01_getting_started.py`
3. **Explore examples**: `docs/examples/`
4. **Check the API reference**: `docs/api_reference.md`

Welcome to PrecipGen! 🌧️