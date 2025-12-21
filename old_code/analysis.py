"""
Module for performing sliding window analysis on precipitation parameters.
"""


import pandas as pd
import numpy as np
from .calculations import calculate_monthly_parameters
import logging
logger = logging.getLogger(__name__)

def analyze_parameter_trends(full_data: pd.DataFrame, window_years: int) -> dict:
    """
    Performs a sliding window analysis to find parameter statistics.

    Args:
        full_data (pd.DataFrame): The complete historical precipitation data.
        window_years (int): The number of years for the sliding window.

    Returns:
        dict: A dictionary with volatility and correlation matrix stats.
    """
    try:
        logger.info(f"Starting sliding window analysis with window_years={window_years}")
        years = full_data.index.year
        first_year = years.min()
        last_year = years.max()
        window_starts = list(range(first_year, last_year - window_years + 2))

        # Define seasons: DJF, MAM, JJA, SON
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Fall', 10: 'Fall', 11: 'Fall'}
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        params = ['PWW', 'PWD', 'ALPHA', 'BETA']

        # Store seasonal parameter time series for each window
        seasonal_records = {season: {p: [] for p in params} for season in seasons}
        window_labels = []
        annual_records = []

        for start in window_starts:
            end = start + window_years - 1
            window_df = full_data[(full_data.index.year >= start) & (full_data.index.year <= end)]
            if window_df.empty or window_df.index.year.nunique() < window_years:
                continue  # skip incomplete windows
            monthly = calculate_monthly_parameters(window_df)
            # Compute seasonal means for each parameter
            season_months = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11]
            }
            season_vals = {season: {} for season in seasons}
            for season, months in season_months.items():
                vals = {p: [] for p in params}
                for m in months:
                    vals['PWW'].append(monthly[m]['p_ww'])
                    vals['PWD'].append(monthly[m]['p_wd'])
                    vals['ALPHA'].append(monthly[m]['alpha'])
                    vals['BETA'].append(monthly[m]['beta'])
                for p in params:
                    mean_val = np.nanmean(vals[p])
                    seasonal_records[season][p].append(mean_val)
                    season_vals[season][p] = mean_val
            # For annual correlation, use annual means
            annual_records.append({p: np.nanmean([season_vals[s][p] for s in seasons]) for p in params})
            window_labels.append(start)

        # Build DataFrames for seasonal stats
        seasonal_df = {season: pd.DataFrame({p: seasonal_records[season][p] for p in params}, index=window_labels) for season in seasons}

        # Calculate reversion and volatility for each season/parameter
        def ar1_reversion(series):
            x = pd.Series(series).dropna().values
            if len(x) < 2:
                return np.nan
            x1 = x[:-1]
            x2 = x[1:]
            if np.std(x1) == 0:
                return np.nan
            ar1 = np.corrcoef(x1, x2)[0, 1]
            return 1 - ar1

        reversion_table = pd.DataFrame(index=seasons, columns=params)
        volatility_table = pd.DataFrame(index=seasons, columns=params)
        trend_table = pd.DataFrame(index=seasons, columns=params)
        for season in seasons:
            for p in params:
                s = seasonal_df[season][p]
                reversion_table.loc[season, p] = ar1_reversion(s)
                volatility_table.loc[season, p] = np.nanstd(s)
                # Trend slope (simple linear fit)
                if len(s.dropna()) > 1:
                    x = np.arange(len(s))
                    y = s.values
                    mask = ~np.isnan(y)
                    if mask.sum() > 1:
                        slope = np.polyfit(x[mask], y[mask], 1)[0]
                        trend_table.loc[season, p] = slope
                    else:
                        trend_table.loc[season, p] = np.nan
                else:
                    trend_table.loc[season, p] = np.nan

        # Annual correlation matrix (from annual means)
        annual_df = pd.DataFrame(annual_records, index=window_labels)
        correlation_matrix = annual_df.corr()

        logger.info("Sliding window analysis complete.")
        return {
            'seasonal_reversion': reversion_table.astype(float),
            'seasonal_volatility': volatility_table.astype(float),
            'seasonal_trend': trend_table.astype(float),
            'correlation_matrix': correlation_matrix,
        }
    except Exception as e:
        logger.exception(f"Error in analyze_parameter_trends: {e}")
        raise
