# Contributing to PrecipGen

Thank you for your interest in contributing to PrecipGen! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setting up Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/precipgen/precipgen.git
   cd precipgen
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Code Style and Standards

### Python Code Style

- Follow PEP 8 style guidelines
- Use type hints for all public functions
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Formatting

We use automated code formatting tools:

```bash
# Format code with Black
black precipgen/ tests/

# Sort imports with isort
isort precipgen/ tests/

# Check code style with flake8
flake8 precipgen/ tests/
```

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples for complex functions
- Keep documentation up-to-date with code changes

Example docstring:
```python
def calculate_transition_probabilities(data: pd.Series, threshold: float = 0.001) -> Tuple[float, float]:
    """
    Calculate Markov chain transition probabilities for precipitation.
    
    Args:
        data: Daily precipitation time series
        threshold: Wet day threshold in same units as data
        
    Returns:
        Tuple of (P(W|W), P(W|D)) transition probabilities
        
    Raises:
        InsufficientDataError: If data has fewer than 30 transitions
        
    Example:
        >>> precip_data = pd.Series([0.0, 2.5, 1.0, 0.0])
        >>> p_ww, p_wd = calculate_transition_probabilities(precip_data)
        >>> print(f"P(W|W): {p_ww:.3f}, P(W|D): {p_wd:.3f}")
    """
```

## Testing Guidelines

### Test Structure

- Unit tests in `tests/` directory
- Property-based tests using Hypothesis
- Integration tests for complete workflows
- Test files named `test_*.py`

### Writing Tests

1. **Unit Tests**: Test individual functions and methods
   ```python
   def test_gamma_parameter_estimation():
       """Test Gamma parameter estimation with known data."""
       # Known data with expected parameters
       wet_days = np.random.gamma(2.0, 3.0, 1000)
       alpha, beta = estimate_gamma_parameters(wet_days)
       
       assert abs(alpha - 2.0) < 0.1
       assert abs(beta - 3.0) < 0.3
   ```

2. **Property-Based Tests**: Test universal properties
   ```python
   @given(precipitation_data=precipitation_time_series())
   @settings(max_examples=100)
   def test_parameter_bounds(precipitation_data):
       """Test that estimated parameters are within valid bounds."""
       p_ww, p_wd = calculate_transition_probabilities(precipitation_data)
       
       assert 0 <= p_ww <= 1
       assert 0 <= p_wd <= 1
   ```

3. **Integration Tests**: Test complete workflows
   ```python
   def test_complete_analysis_workflow():
       """Test complete analysis from data to parameters."""
       # Load test data
       data = load_test_precipitation_data()
       
       # Run complete workflow
       engine = AnalyticalEngine(data)
       manifest = engine.generate_parameter_manifest()
       
       # Verify manifest structure
       assert 'metadata' in manifest
       assert 'overall_parameters' in manifest
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=precipgen

# Run specific test file
pytest tests/test_analytical_engine.py

# Run property-based tests with more examples
pytest tests/test_properties.py --hypothesis-max-examples=1000
```

## Mathematical Accuracy

### Algorithm Implementation

- Implement algorithms exactly as described in Richardson & Wright (1984)
- Include references to equations in code comments
- Validate against published results when possible
- Handle edge cases gracefully

### Numerical Considerations

- Use numerically stable algorithms
- Handle floating-point precision issues
- Validate parameter bounds
- Provide clear error messages for invalid inputs

Example:
```python
def estimate_gamma_parameters(wet_day_amounts: pd.Series) -> Tuple[float, float]:
    """
    Estimate Gamma distribution parameters using method of moments.
    
    Following Richardson & Wright (1984), equations 3-4:
    α = μ² / σ²  (shape parameter)
    β = σ² / μ   (scale parameter)
    """
    if len(wet_day_amounts) < 10:
        raise InsufficientDataError(
            f"Need at least 10 wet days for reliable estimation, got {len(wet_day_amounts)}"
        )
    
    mean_precip = wet_day_amounts.mean()
    var_precip = wet_day_amounts.var(ddof=1)
    
    if var_precip <= 0 or mean_precip <= 0:
        raise EstimationError(
            f"Invalid sample statistics: mean={mean_precip:.3f}, var={var_precip:.3f}"
        )
    
    # Method of moments estimators
    alpha = (mean_precip ** 2) / var_precip
    beta = var_precip / mean_precip
    
    # Validate results
    if alpha <= 0 or beta <= 0:
        raise EstimationError(
            f"Estimated parameters not positive: α={alpha:.3f}, β={beta:.3f}"
        )
    
    return alpha, beta
```

## Contribution Process

### 1. Issue Discussion

- Check existing issues before creating new ones
- Use issue templates for bug reports and feature requests
- Discuss major changes before implementation
- Tag issues appropriately (bug, enhancement, documentation, etc.)

### 2. Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test changes**
   ```bash
   pytest
   black precipgen/ tests/
   flake8 precipgen/ tests/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### 3. Pull Request Guidelines

- Use descriptive PR titles and descriptions
- Reference related issues
- Include test results
- Update CHANGELOG.md for significant changes
- Ensure CI passes

### 4. Code Review Process

- All PRs require review from maintainers
- Address review comments promptly
- Maintain respectful discussion
- Be open to suggestions and improvements

## Types of Contributions

### Bug Fixes

- Fix incorrect algorithm implementations
- Handle edge cases properly
- Improve error messages
- Fix documentation errors

### New Features

- Additional analysis methods
- New data source formats
- Performance improvements
- Enhanced visualization tools

### Documentation

- API documentation improvements
- Tutorial enhancements
- Example code additions
- Mathematical documentation

### Testing

- Additional unit tests
- Property-based test improvements
- Integration test coverage
- Performance benchmarks

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release tag
6. Build and upload to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: precipgen-dev@example.com for private matters

### Documentation

- **User Guide**: `docs/user_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Examples**: `docs/examples/`
- **Mathematical Foundation**: `docs/mathematical_foundation.md`

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments

Thank you for contributing to PrecipGen! Your efforts help advance open-source climate modeling tools.