# Algorithm Documentation

## Overview

This document provides detailed descriptions of all algorithms implemented in PrecipGen, including their assumptions, limitations, and implementation details.

## Data Processing Algorithms

### 1. GHCN Data Parser

#### Purpose
Parse GHCN Daily (.dly) format files and extract precipitation data with quality assessment.

#### Algorithm Description
```python
def parse_ghcn_dly(filepath):
    """
    Parse GHCN .dly file format
    
    Format specification:
    - Positions 1-11: Station ID
    - Positions 12-15: Year
    - Positions 16-17: Month  
    - Positions 18-21: Element (e.g., PRCP)
    - Positions 22-269: 31 daily values (8 chars each)
    """
    
    for line in file:
        station_id = line[0:11]
        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21]
        
        if element == 'PRCP':
            for day in range(31):
                start_pos = 21 + day * 8
                value = line[start_pos:start_pos+5]
                flag = line[start_pos+5:start_pos+6]
                
                if value != '-9999':  # Missing data indicator
                    # Convert from tenths of mm to mm
                    precip_mm = int(value) / 10.0
                    yield (year, month, day+1, precip_mm, flag)
```

#### Assumptions
- Input files follow GHCN Daily format specification exactly
- Missing data is indicated by -9999
- Precipitation values are in tenths of millimeters
- Quality flags follow GHCN standards

#### Error Handling
- Invalid file format raises `ParseError`
- Missing files raise `FileNotFoundError`
- Corrupted data lines are logged and skipped

### 2. Data Validation Algorithm

#### Purpose
Assess data quality and completeness for reliable parameter estimation.

#### Algorithm Description
```python
def validate_data_quality(data, thresholds):
    """
    Assess data quality based on configurable thresholds
    
    Quality metrics:
    - Completeness: fraction of non-missing values
    - Physical bounds: precipitation values within reasonable limits
    - Quality flags: assessment of GHCN quality indicators
    """
    
    total_days = len(data)
    missing_days = data.isna().sum()
    completeness = 1.0 - (missing_days / total_days)
    
    # Physical bounds check
    valid_range = (data >= 0) & (data <= 1000)  # 0-1000mm reasonable
    out_of_bounds = (~valid_range).sum()
    
    # Quality assessment
    quality_score = completeness * (1.0 - out_of_bounds / total_days)
    
    return QualityReport(
        completeness=completeness,
        quality_score=quality_score,
        out_of_bounds_count=out_of_bounds,
        recommendation="ACCEPT" if quality_score > thresholds.min_quality else "REJECT"
    )
```

#### Assumptions
- Precipitation values should be non-negative
- Values > 1000mm/day are considered suspicious
- Completeness threshold typically 80-90%

## Parameter Estimation Algorithms

### 3. Transition Probability Calculation

#### Purpose
Calculate P(W|W) and P(W|D) from historical precipitation data.

#### Algorithm Description
```python
def calculate_transition_probabilities(precip_data, wet_threshold=0.001):
    """
    Calculate Markov chain transition probabilities
    
    States:
    - Wet: precipitation >= wet_threshold
    - Dry: precipitation < wet_threshold
    """
    
    # Convert to wet/dry states
    wet_days = precip_data >= wet_threshold
    
    # Count transitions
    ww_count = 0  # wet following wet
    wd_count = 0  # wet following dry
    dw_count = 0  # dry following wet
    dd_count = 0  # dry following dry
    
    for i in range(1, len(wet_days)):
        prev_wet = wet_days.iloc[i-1]
        curr_wet = wet_days.iloc[i]
        
        if prev_wet and curr_wet:
            ww_count += 1
        elif prev_wet and not curr_wet:
            dw_count += 1
        elif not prev_wet and curr_wet:
            wd_count += 1
        else:
            dd_count += 1
    
    # Calculate probabilities
    total_wet_prev = ww_count + dw_count
    total_dry_prev = wd_count + dd_count
    
    p_ww = ww_count / total_wet_prev if total_wet_prev > 0 else 0.0
    p_wd = wd_count / total_dry_prev if total_dry_prev > 0 else 0.0
    
    return p_ww, p_wd
```

#### Assumptions
- First-order Markov chain (current state depends only on previous day)
- Wet day threshold is appropriate for the climate region
- Sufficient data for reliable estimation (typically n ≥ 100 transitions)

#### Error Handling
- Insufficient data returns NaN with warning
- All dry or all wet periods handled gracefully

### 4. Gamma Parameter Estimation

#### Purpose
Estimate shape (α) and scale (β) parameters for wet-day precipitation amounts.

#### Algorithm Description
```python
def estimate_gamma_parameters(wet_day_amounts):
    """
    Estimate Gamma distribution parameters using method of moments
    
    Method of moments:
    - α = μ² / σ²  (shape parameter)
    - β = σ² / μ   (scale parameter)
    
    Where μ = sample mean, σ² = sample variance
    """
    
    if len(wet_day_amounts) < 10:
        raise InsufficientDataError("Need at least 10 wet days for reliable estimation")
    
    # Calculate sample statistics
    mean_precip = wet_day_amounts.mean()
    var_precip = wet_day_amounts.var(ddof=1)  # Sample variance
    
    if var_precip <= 0 or mean_precip <= 0:
        raise EstimationError("Invalid sample statistics for Gamma fitting")
    
    # Method of moments estimators
    alpha = (mean_precip ** 2) / var_precip
    beta = var_precip / mean_precip
    
    # Validate parameters
    if alpha <= 0 or beta <= 0:
        raise EstimationError("Estimated parameters are not positive")
    
    return alpha, beta
```

#### Assumptions
- Wet-day amounts follow Gamma distribution
- Method of moments provides reasonable parameter estimates
- Minimum sample size for reliable estimation

#### Alternative Methods
- Maximum likelihood estimation (more accurate but computationally intensive)
- L-moments method (robust to outliers)

## Trend Analysis Algorithms

### 5. Sliding Window Analysis

#### Purpose
Analyze temporal evolution of parameters using overlapping time windows.

#### Algorithm Description
```python
def sliding_window_analysis(data, window_years=30, step_years=1):
    """
    Perform sliding window analysis of precipitation parameters
    
    Process:
    1. Define overlapping windows of specified size
    2. Calculate parameters for each window
    3. Track parameter evolution over time
    """
    
    results = []
    start_year = data.index.year.min()
    end_year = data.index.year.max()
    
    for center_year in range(start_year + window_years//2, 
                           end_year - window_years//2 + 1, 
                           step_years):
        
        window_start = center_year - window_years // 2
        window_end = center_year + window_years // 2
        
        # Extract window data
        window_data = data[
            (data.index.year >= window_start) & 
            (data.index.year <= window_end)
        ]
        
        if len(window_data) < window_years * 300:  # Minimum data requirement
            continue
            
        # Calculate parameters for this window
        monthly_params = {}
        for month in range(1, 13):
            month_data = window_data[window_data.index.month == month]
            
            if len(month_data) >= 50:  # Minimum monthly data
                p_ww, p_wd = calculate_transition_probabilities(month_data)
                wet_amounts = month_data[month_data >= wet_threshold]
                
                if len(wet_amounts) >= 10:
                    alpha, beta = estimate_gamma_parameters(wet_amounts)
                    monthly_params[month] = {
                        'p_ww': p_ww, 'p_wd': p_wd,
                        'alpha': alpha, 'beta': beta
                    }
        
        results.append({
            'center_year': center_year,
            'parameters': monthly_params
        })
    
    return results
```

#### Assumptions
- Window size captures sufficient climate variability
- Overlapping windows provide smooth parameter evolution
- Linear trends are appropriate for the analysis period

### 6. Trend Detection Algorithm

#### Purpose
Detect statistically significant trends in parameter evolution.

#### Algorithm Description
```python
def detect_trends(window_results, significance_level=0.05):
    """
    Detect linear trends in parameter evolution using regression analysis
    
    Process:
    1. Extract parameter time series from window results
    2. Fit linear regression for each parameter
    3. Test statistical significance of trends
    """
    
    trends = {}
    
    # Extract time series for each parameter
    for param_name in ['p_ww', 'p_wd', 'alpha', 'beta']:
        for month in range(1, 13):
            
            # Collect parameter values over time
            years = []
            values = []
            
            for result in window_results:
                if month in result['parameters']:
                    years.append(result['center_year'])
                    values.append(result['parameters'][month][param_name])
            
            if len(values) < 10:  # Minimum points for trend analysis
                continue
            
            # Fit linear regression
            years = np.array(years)
            values = np.array(values)
            
            # Center years for numerical stability
            years_centered = years - years.mean()
            
            # Calculate regression coefficients
            slope = np.sum(years_centered * values) / np.sum(years_centered ** 2)
            intercept = values.mean()
            
            # Calculate residuals and statistics
            predicted = intercept + slope * years_centered
            residuals = values - predicted
            mse = np.sum(residuals ** 2) / (len(values) - 2)
            
            # Standard error of slope
            se_slope = np.sqrt(mse / np.sum(years_centered ** 2))
            
            # t-statistic and p-value
            t_stat = slope / se_slope if se_slope > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(values) - 2))
            
            # Store trend information
            key = f"{param_name}_month_{month}"
            trends[key] = {
                'slope': slope,
                'intercept': intercept,
                'p_value': p_value,
                'significant': p_value < significance_level,
                'r_squared': 1 - np.sum(residuals ** 2) / np.sum((values - values.mean()) ** 2)
            }
    
    return trends
```

#### Assumptions
- Linear trends are appropriate for the time period
- Residuals are normally distributed
- Independence of observations (may be violated with overlapping windows)

## Simulation Algorithms

### 7. Bootstrap Resampling

#### Purpose
Generate synthetic precipitation by resampling historical years.

#### Algorithm Description
```python
def bootstrap_simulation(historical_data, mode='random', random_seed=None):
    """
    Generate precipitation using bootstrap resampling
    
    Modes:
    - 'random': Randomly select historical years with replacement
    - 'sequential': Cycle through years in chronological order
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Organize data by year
    years_data = {}
    for year in historical_data.index.year.unique():
        year_data = historical_data[historical_data.index.year == year]
        years_data[year] = year_data
    
    available_years = list(years_data.keys())
    current_year_idx = 0
    
    def step():
        nonlocal current_year_idx
        
        if mode == 'random':
            selected_year = np.random.choice(available_years)
        elif mode == 'sequential':
            selected_year = available_years[current_year_idx]
            current_year_idx = (current_year_idx + 1) % len(available_years)
        
        # Return year's data for sequential access
        return years_data[selected_year]
    
    return step
```

#### Assumptions
- Historical years are representative of future conditions
- Year-to-year transitions are handled appropriately
- Leap year differences are managed correctly

### 8. WGEN Simulation Algorithm

#### Purpose
Generate synthetic precipitation using Markov chain and Gamma distribution.

#### Algorithm Description
```python
def wgen_simulation(parameters, start_date, trend_mode=False, random_seed=None):
    """
    Generate synthetic precipitation using WGEN algorithm
    
    Process:
    1. Initialize state (wet/dry, date, random generator)
    2. For each day:
       a. Determine current month parameters
       b. Apply trend adjustment if enabled
       c. Generate wet/dry state using Markov chain
       d. Generate precipitation amount if wet day
       e. Update state for next day
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize state
    current_date = start_date
    is_wet_yesterday = False  # Initial state
    
    def step():
        nonlocal current_date, is_wet_yesterday
        
        # Get current month parameters
        month = current_date.month
        base_params = parameters['monthly'][month]
        
        # Apply trend adjustment if enabled
        if trend_mode and 'trends' in parameters:
            elapsed_years = (current_date - start_date).days / 365.25
            
            # Adjust parameters based on trends
            p_ww = base_params['p_ww'] + parameters['trends']['p_ww'][month] * elapsed_years
            p_wd = base_params['p_wd'] + parameters['trends']['p_wd'][month] * elapsed_years
            alpha = base_params['alpha'] + parameters['trends']['alpha'][month] * elapsed_years
            beta = base_params['beta'] + parameters['trends']['beta'][month] * elapsed_years
            
            # Apply physical bounds
            p_ww = max(0.0, min(1.0, p_ww))
            p_wd = max(0.0, min(1.0, p_wd))
            alpha = max(0.001, alpha)
            beta = max(0.001, beta)
        else:
            p_ww = base_params['p_ww']
            p_wd = base_params['p_wd']
            alpha = base_params['alpha']
            beta = base_params['beta']
        
        # Determine wet/dry state using Markov chain
        if is_wet_yesterday:
            prob_wet_today = p_ww
        else:
            prob_wet_today = p_wd
        
        is_wet_today = np.random.random() < prob_wet_today
        
        # Generate precipitation amount
        if is_wet_today:
            # Sample from Gamma distribution
            precipitation = np.random.gamma(alpha, beta)
        else:
            precipitation = 0.0
        
        # Update state for next iteration
        is_wet_yesterday = is_wet_today
        current_date += timedelta(days=1)
        
        return precipitation
    
    return step
```

#### Assumptions
- First-order Markov chain adequately models wet/dry persistence
- Gamma distribution appropriately models wet-day amounts
- Monthly stratification captures seasonal patterns
- Linear trend adjustment is appropriate for non-stationary simulation

#### Error Handling
- Invalid parameters raise `ParameterError`
- Trend adjustments that violate bounds are constrained
- Random number generator state is managed properly

## Quality Control Algorithms

### 9. Parameter Validation

#### Purpose
Validate estimated parameters for physical and statistical reasonableness.

#### Algorithm Description
```python
def validate_parameters(parameters):
    """
    Validate parameter estimates for physical and statistical reasonableness
    
    Checks:
    - Transition probabilities in [0, 1]
    - Gamma parameters positive
    - Reasonable parameter ranges
    - Sufficient sample sizes
    """
    
    validation_results = []
    
    for month, params in parameters.items():
        # Check transition probabilities
        if not (0 <= params['p_ww'] <= 1):
            validation_results.append(f"Month {month}: P(W|W) = {params['p_ww']} outside [0,1]")
        
        if not (0 <= params['p_wd'] <= 1):
            validation_results.append(f"Month {month}: P(W|D) = {params['p_wd']} outside [0,1]")
        
        # Check Gamma parameters
        if params['alpha'] <= 0:
            validation_results.append(f"Month {month}: Alpha = {params['alpha']} not positive")
        
        if params['beta'] <= 0:
            validation_results.append(f"Month {month}: Beta = {params['beta']} not positive")
        
        # Check reasonable ranges (climate-dependent)
        if params['alpha'] > 10:
            validation_results.append(f"Month {month}: Alpha = {params['alpha']} unusually large")
        
        if params['beta'] > 100:
            validation_results.append(f"Month {month}: Beta = {params['beta']} unusually large")
    
    return validation_results
```

### 10. Statistical Testing

#### Purpose
Perform goodness-of-fit tests to validate distributional assumptions.

#### Algorithm Description
```python
def goodness_of_fit_tests(observed_data, fitted_parameters):
    """
    Perform statistical tests to validate model assumptions
    
    Tests:
    - Kolmogorov-Smirnov test for Gamma distribution
    - Chi-square test for transition probabilities
    - Anderson-Darling test for normality of residuals
    """
    
    from scipy import stats
    
    test_results = {}
    
    # Test Gamma distribution fit for wet days
    wet_days = observed_data[observed_data > 0]
    alpha, beta = fitted_parameters['alpha'], fitted_parameters['beta']
    
    # Generate theoretical quantiles
    theoretical_quantiles = stats.gamma.ppf(
        np.linspace(0.01, 0.99, len(wet_days)), 
        alpha, scale=beta
    )
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = stats.kstest(
        wet_days, 
        lambda x: stats.gamma.cdf(x, alpha, scale=beta)
    )
    
    test_results['gamma_ks'] = {
        'statistic': ks_statistic,
        'p_value': ks_p_value,
        'reject_null': ks_p_value < 0.05
    }
    
    # Anderson-Darling test
    ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(
        wet_days, 
        dist='gamma'
    )
    
    test_results['gamma_ad'] = {
        'statistic': ad_statistic,
        'critical_values': ad_critical_values,
        'significance_levels': ad_significance_levels
    }
    
    return test_results
```

## Performance Considerations

### Memory Management
- Use generators for large datasets to minimize memory usage
- Implement efficient data structures for time series operations
- Cache frequently accessed parameters

### Computational Efficiency
- Vectorize operations using NumPy where possible
- Pre-compute monthly parameters to avoid repeated calculations
- Use efficient random number generation

### Numerical Stability
- Handle edge cases in parameter estimation
- Use numerically stable algorithms for regression analysis
- Implement proper error handling for degenerate cases