---
## Part 3: Roadmap Feature — Seasonal Trend Analysis (Planned for v2.0)

### The Core Idea: Pattern vs. Trend
A seasonal pattern is something we already know. For example, in many places, winter is wetter than summer. This is the baseline, average behavior.

A seasonal trend asks a deeper question: Is this year's winter wetter or drier on average than winters were 50 years ago?

The goal of this new feature is to automatically detect and measure those long-term changes within each season.

#### Why It's Important
This analysis helps us answer critical questions for long-term planning, such as:

* "Are our springs getting progressively drier over the decades?"
* "Is summer rainfall becoming more intense but less frequent?"

Instead of just having a single, static average for "winter," we'd be able to see the direction winter precipitation is heading over time.

#### How It Works (The Simple Version)
Our plan to measure this is straightforward:

1. **Group Data:** We'd look at all the winters in the historical record as one group, all the springs as another, and so on.
2. **Calculate Averages:** For each individual year, we'd calculate the average parameters (PWW, PWD, etc.) for that year's winter. This gives us a time series of "winter characteristics."
3. **Find the Trend:** We'd then plot that time series (e.g., the "Winter PWW" for 1950, 1951, 1952...) and find the best-fit line through the points. The slope of that line is the trend. A positive slope means that characteristic is increasing over time; a negative slope means it's decreasing.

---
## Part 1: Mission Brief (Lead Dev Context)

This section outlines the high-level strategic vision for the project.

### Project Goal
A minimal, professional Python tool to calculate stochastic precipitation parameters (`P(W|W)`, `P(W|D)`, `alpha`, `beta`) and their statistical trends from a daily precipitation time series in a csv file for use with other simulation tools like Hydrosim or GoldSim.

### Core User Story
As a user, I want to provide an csv file containing historical daily precipitation data (often with GHCN metadata headers) and receive a clean output of monthly PrecipGen parameters and their statistical trends (volatility, reversion rate, correlations) to use directly in my simulations.

### Key Design Principles
1.  **csv-Centric Workflow:** The primary input and output format is Excel to match user workflows.
2.  **Robust Data Parsing:** Automatically handle metadata headers in the input file.
3.  **Minimal Dependencies:** Use a small, standard set of libraries for easy installation.
4.  **Professional Code Quality:** Enforce code formatting (`black`), linting (`ruff`), and unit testing (`pytest`).
5.  **Single Source of Truth:** This document guides all development.

### Development Stack
* **Language:** Python 3.9+
* **Environment:** `venv`
* **Core Libraries:** `pandas`, `openpyxl`, `scipy`
* **Dev Tools:** `pytest`, `black`, `ruff`

---
## Part 2: Implementation Blueprint (Junior Dev / Copilot Context)

This section provides detailed technical specifications for implementation.

### File Structure
````
/precip_param_analyzer/
|
├── .gitignore
├── requirements.txt
├── README.md
├── PROJECT_CONTEXT.md
|
├── main.py             # User entry point
|
└── src/                # Source code module
|   ├── data_loader.py    # Reads and validates Excel data
|   ├── calculations.py   # Calculates base PrecipGen parameters
|   ├── analysis.py       # Runs sliding window analysis
|   └── output_generator.py # Writes results back to Excel
|
└── tests/              # Unit tests
├── test_data_loader.py
└── test_calculations.py
````

### Module Responsibilities
* **`main.py`**: Orchestrates the entire process. Calls functions from the `src` modules in sequence.
* **`src/data_loader.py`**: Contains a function that accepts an Excel file path, finds the `DATE`/`PRCP` header, and returns a clean pandas DataFrame.
* **`src/calculations.py`**: Contains functions to calculate `P(W|W)`, `P(W|D)`, `alpha`, and `beta` from a DataFrame.
* **`src/analysis.py`**: Contains functions to perform the sliding window analysis, calculating volatility, reversion rates, and correlations of the parameters.
* **`src/output_generator.py`**: Contains a function to take the final results and write them to a new sheet in the original Excel file.
