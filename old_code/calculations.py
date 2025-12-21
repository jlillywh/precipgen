"""
Module for calculating the core WGEN precipitation parameters.

Acknowledgment:
This tool implements the precipitation parameterization and gamma fitting methods from the original WGEN weather generator (Richardson & Wright, 1984; Allen Wright, 1983). The monthly parameter calculations and formulas are directly based on the published WGEN Fortran code and GoldSim’s implementation.
"""


import pandas as pd
import numpy as np
from scipy.stats import gamma
import logging
logger = logging.getLogger(__name__)

def calculate_monthly_parameters(data: pd.DataFrame, wet_day_threshold: float = 0.1) -> dict:
    """
    Calculates P(W|W), P(W|D), alpha, and beta for each month.

    Args:
        data (pd.DataFrame): DataFrame with precipitation data. Must have a DatetimeIndex and 'precip' column.
        wet_day_threshold (float): The minimum precipitation value to be considered a "wet day".

    Returns:
        dict: A dictionary containing the parameters for each month.
    """
    # Accepts: data (with DatetimeIndex and 'precip'), wet_day_threshold (float)

    try:
        logger.debug(f"Calculating monthly parameters with wet_day_threshold={wet_day_threshold}")
        df = data.copy()
        df['is_wet'] = df['precip'] >= wet_day_threshold
        df['prev_is_wet'] = df['is_wet'].shift(1)

        results = {}
        for month in range(1, 13):
            month_df = df[df.index.month == month]
            month_df = month_df[~month_df['prev_is_wet'].isna()]
            ww = ((month_df['is_wet'] == True) & (month_df['prev_is_wet'] == True)).sum()
            wd = ((month_df['is_wet'] == True) & (month_df['prev_is_wet'] == False)).sum()
            dw = ((month_df['is_wet'] == False) & (month_df['prev_is_wet'] == True)).sum()
            dd = ((month_df['is_wet'] == False) & (month_df['prev_is_wet'] == False)).sum()
            ww_denom = ((month_df['prev_is_wet'] == True)).sum()
            wd_denom = ((month_df['prev_is_wet'] == False)).sum()
            try:
                p_ww = ww / ww_denom if ww_denom > 0 else np.nan
            except Exception as e:
                logger.warning(f"Error calculating P(W|W) for month {month}: {e}")
                p_ww = np.nan
            try:
                p_wd = wd / wd_denom if wd_denom > 0 else np.nan
            except Exception as e:
                logger.warning(f"Error calculating P(W|D) for month {month}: {e}")
                p_wd = np.nan

            wet_precip = month_df.loc[month_df['is_wet'], 'precip']
            if wet_precip.count() >= 3:
                rbar = wet_precip.mean()
                rlbar = np.log(wet_precip).mean()
                y = np.log(rbar) - rlbar if rbar > 0 else np.nan
                if y is not None and not np.isnan(y) and y > 0:
                    anum = 8.898919 + 9.05995 * y + 0.9775373 * y * y
                    adom = y * (17.79728 + 11.968477 * y + y * y)
                    alpha = anum / adom if adom != 0 else np.nan
                    if alpha >= 1.0:
                        alpha = 0.998
                    beta = rbar / alpha if alpha > 0 else np.nan
                else:
                    alpha, beta = np.nan, np.nan
            else:
                alpha, beta = np.nan, np.nan
            results[month] = {
                'p_ww': float(p_ww),
                'p_wd': float(p_wd),
                'alpha': float(alpha) if not np.isnan(alpha) else np.nan,
                'beta': float(beta) if not np.isnan(beta) else np.nan,
            }
        logger.debug("Monthly parameter calculation complete.")
        return results
    except Exception as e:
        logger.exception(f"Error in calculate_monthly_parameters: {e}")
        raise
