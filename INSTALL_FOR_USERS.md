# Quick Installation Guide for PrecipGen

## For End Users

### Step 1: Install PrecipGen

Open your terminal/command prompt and run:

```bash
pip install git+https://github.com/jlillywh/precipgen.git
```

### Step 2: Verify Installation

Create a new Python file or Jupyter notebook and test:

```python
import precipgen as pg
print(f"PrecipGen version: {pg.__version__}")
print("✓ Installation successful!")
```

### Step 3: Quick Test

Run this complete example:

```python
import precipgen as pg
import numpy as np
import pandas as pd
from datetime import datetime

# Generate test data
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
precip = np.random.gamma(1.5, 3.0, len(dates)) * (np.random.random(len(dates)) < 0.3)
data = pd.Series(precip, index=dates)

# Analyze historical data
engine = pg.AnalyticalEngine(data, wet_day_threshold=0.001)
manifest = engine.generate_parameter_manifest()

# Generate synthetic precipitation
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))
synthetic = [sim.step() for _ in range(365)]

print(f"Generated {len(synthetic)} days of synthetic precipitation")
print(f"Annual total: {sum(synthetic):.1f} mm")
```

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'precipgen'"

**Solution 1 - Check your environment:**
```bash
# Make sure you're in the right Python environment
python -c "import sys; print(sys.executable)"
pip list | grep precipgen
```

**Solution 2 - Reinstall:**
```bash
pip uninstall precipgen
pip install git+https://github.com/jlillywh/precipgen.git
```

**Solution 3 - For Jupyter notebooks:**
```bash
# Install in the correct kernel
pip install ipykernel
python -m ipykernel install --user --name=precipgen-env
# Then select this kernel in your notebook
```

### Problem: Missing dependencies

**Solution:**
```bash
pip install numpy pandas scipy matplotlib pyyaml
pip install git+https://github.com/jlillywh/precipgen.git
```

### Problem: Permission errors

**Solution:**
```bash
# Install for current user only
pip install --user git+https://github.com/jlillywh/precipgen.git
```

## Using Virtual Environments (Recommended)

```bash
# Create virtual environment
python -m venv precipgen_env

# Activate it
# Windows:
precipgen_env\Scripts\activate
# Mac/Linux:
source precipgen_env/bin/activate

# Install PrecipGen
pip install git+https://github.com/jlillywh/precipgen.git

# Test installation
python -c "import precipgen; print('Success!')"
```

## Need Help?

1. **Run the test script:** Download and run `test_installation.py` from the repository
2. **Check the full documentation:** See `docs/installation.md` for detailed instructions
3. **Report issues:** https://github.com/jlillywh/precipgen/issues

## Next Steps

- **Examples:** Check out `docs/examples/` for working code examples
- **User Guide:** Read `docs/user_guide.md` for detailed usage instructions  
- **API Reference:** See `docs/api_reference.md` for complete documentation