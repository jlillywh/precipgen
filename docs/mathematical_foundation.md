# Mathematical Foundation

## Overview

PrecipGen implements the Richardson & Wright (1984) weather generation methodology with modern enhancements for non-stationary climate modeling. This document provides the mathematical foundation for all algorithms implemented in the library.

## References

**Primary Reference:**
Richardson, C.W., and Wright, D.A. (1984). WGEN: A model for generating daily weather variables. U.S. Department of Agriculture, Agricultural Research Service, ARS-8.

**Supporting References:**
- Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences. Academic Press.
- Semenov, M.A., and Barrow, E.M. (1997). Use of a stochastic weather generator in the development of climate change scenarios. Climatic Change, 35(4), 397-414.

## Core Mathematical Framework

### 1. Markov Chain Model for Wet/Dry States

The fundamental assumption is that precipitation occurrence follows a first-order Markov chain where the probability of precipitation on day *t* depends only on the state of day *t-1*.

#### State Definitions
- **Wet day**: Daily precipitation ≥ threshold (typically 0.001 inches or 0.0254 mm)
- **Dry day**: Daily precipitation < threshold

#### Transition Probabilities

**P(W|W)**: Probability of a wet day following a wet day
```
P(W|W) = N(WW) / N(W)
```

**P(W|D)**: Probability of a wet day following a dry day  
```
P(W|D) = N(WD) / N(D)
```

Where:
- N(WW) = number of wet days following wet days
- N(WD) = number of wet days following dry days  
- N(W) = total number of wet days
- N(D) = total number of dry days

#### Mathematical Properties
- 0 ≤ P(W|W) ≤ 1
- 0 ≤ P(W|D) ≤ 1
- P(D|W) = 1 - P(W|W)
- P(D|D) = 1 - P(W|D)

### 2. Gamma Distribution for Precipitation Amounts

On wet days, precipitation amounts follow a two-parameter Gamma distribution:

#### Probability Density Function
```
f(x; α, β) = (1/Γ(α)) * (1/β)^α * x^(α-1) * exp(-x/β)
```

Where:
- x = daily precipitation amount (mm)
- α = shape parameter (α > 0)
- β = scale parameter (β > 0)
- Γ(α) = gamma function

#### Parameter Estimation (Method of Moments)

**Sample mean**: μ = Σx_i / n

**Sample variance**: σ² = Σ(x_i - μ)² / (n-1)

**Shape parameter**: α = μ² / σ²

**Scale parameter**: β = σ² / μ

#### Mathematical Properties
- Mean: E[X] = αβ
- Variance: Var[X] = αβ²
- Both parameters must be positive: α > 0, β > 0

### 3. Monthly Parameter Stratification

Parameters are calculated separately for each calendar month to capture seasonal patterns:

#### Monthly Transition Probabilities
```
P_m(W|W) = N_m(WW) / N_m(W)
P_m(W|D) = N_m(WD) / N_m(D)
```

#### Monthly Gamma Parameters
```
α_m = μ_m² / σ_m²
β_m = σ_m² / μ_m
```

Where subscript *m* indicates month-specific calculations.

## Non-Stationary Extensions

### 4. Sliding Window Analysis

To detect temporal changes in parameters, we use overlapping sliding windows:

#### Window Definition
- Window size: W years (typically 20-30 years)
- Step size: S years (typically 1 year)
- Total windows: (N - W + 1) where N = total years

#### Parameter Evolution
For each window *i* centered at year *t_i*:
```
P_i(W|W), P_i(W|D), α_i, β_i = f(data[t_i - W/2 : t_i + W/2])
```

### 5. Trend Analysis

#### Linear Trend Model
```
Parameter(t) = β₀ + β₁ * t + ε(t)
```

Where:
- β₀ = intercept (baseline parameter value)
- β₁ = slope (trend slope, units per year)
- t = time (years from reference point)
- ε(t) = residual error

#### Trend Slope Calculation
Using ordinary least squares regression:
```
β₁ = Σ[(t_i - t̄)(P_i - P̄)] / Σ[(t_i - t̄)²]
β₀ = P̄ - β₁ * t̄
```

#### Statistical Significance Testing
**t-statistic**:
```
t = β₁ / SE(β₁)
```

**Standard error**:
```
SE(β₁) = √[MSE / Σ(t_i - t̄)²]
```

**Mean squared error**:
```
MSE = Σ[P_i - (β₀ + β₁ * t_i)]² / (n - 2)
```

### 6. Parameter Drift in Simulation

#### Dynamic Parameter Calculation
During simulation, parameters are adjusted based on elapsed time:

```
P(W|W)(t) = P(W|W)_baseline + slope_PWW * t
P(W|D)(t) = P(W|D)_baseline + slope_PWD * t
α(t) = α_baseline + slope_alpha * t
β(t) = β_baseline + slope_beta * t
```

#### Physical Bounds Enforcement
**Transition probabilities**:
```
P(W|W)(t) = max(0, min(1, P(W|W)(t)))
P(W|D)(t) = max(0, min(1, P(W|D)(t)))
```

**Gamma parameters**:
```
α(t) = max(ε, α(t))  where ε = 0.001
β(t) = max(ε, β(t))  where ε = 0.001
```

## Simulation Algorithm

### 7. Daily Weather Generation

#### Step 1: Determine Wet/Dry State
```
if random() < P(W|previous_state):
    current_state = WET
else:
    current_state = DRY
```

#### Step 2: Generate Precipitation Amount
```
if current_state == WET:
    precipitation = gamma_random(α_month, β_month)
else:
    precipitation = 0.0
```

#### Step 3: Update State
```
previous_state = current_state
current_date = current_date + 1 day
```

## Quality Assurance

### 8. Parameter Validation

#### Transition Probability Checks
- 0 ≤ P(W|W) ≤ 1
- 0 ≤ P(W|D) ≤ 1
- Sufficient sample size for reliable estimation (typically n ≥ 30)

#### Gamma Parameter Checks
- α > 0 (shape parameter must be positive)
- β > 0 (scale parameter must be positive)
- Reasonable parameter ranges based on physical constraints

#### Data Quality Requirements
- Minimum data completeness (typically 80-90%)
- Quality flag assessment for GHCN data
- Outlier detection and handling

### 9. Statistical Validation

#### Goodness-of-Fit Tests
- Kolmogorov-Smirnov test for Gamma distribution fit
- Chi-square test for transition probability validation
- Anderson-Darling test for distribution assessment

#### Long-term Statistics Preservation
Generated sequences should preserve:
- Monthly mean precipitation
- Monthly precipitation variance
- Wet day frequency
- Dry spell length distributions
- Wet spell length distributions

## Implementation Notes

### Numerical Considerations
- Use stable algorithms for Gamma parameter estimation
- Handle edge cases (all dry periods, insufficient data)
- Implement proper random number generator seeding for reproducibility

### Performance Optimization
- Pre-compute monthly parameters for efficiency
- Use vectorized operations where possible
- Implement efficient state management for long simulations

### Error Handling
- Graceful degradation when insufficient data exists
- Clear error messages for parameter estimation failures
- Robust handling of edge cases in trend analysis