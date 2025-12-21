"""
Module for loading and validating precipitation data from an Excel file.
"""



import pandas as pd
import os
import logging
logger = logging.getLogger(__name__)


def load_precipitation_data(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Loads daily precipitation data from a specified sheet in an Excel file.

    This function scans the sheet to automatically find the header row
    containing 'DATE' and 'PRCP' columns, skipping any metadata rows above it.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str, optional): Name of the Excel sheet to read. If None, uses the first sheet.

    Returns:
        pd.DataFrame: A DataFrame with a DatetimeIndex and 'precip' column.
    """
    logger.info(f"Loading precipitation data from {file_path}")
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Find header row by scanning first 50 rows for DATE and PRCP columns
    # Look for various common column name patterns
    raw = pd.read_excel(file_path, sheet_name=sheet_name, nrows=50, header=None, engine="openpyxl")
    header_row = None
    date_col = None
    prcp_col = None
    
    for i, row in raw.iterrows():
        vals = [str(val).strip().upper() for val in row.values if pd.notna(val)]
        
        # Look for date column patterns
        date_found = any(date_pattern in val for val in vals 
                        for date_pattern in ['DATE', 'DAY', 'TIME', 'DT'])
        
        # Look for precipitation column patterns  
        prcp_found = any(prcp_pattern in val for val in vals
                        for prcp_pattern in ['PRCP', 'PRECIP', 'PRECIPITATION', 'RAIN', 'PPT'])
        
        if date_found and prcp_found:
            header_row = i
            # Find the exact column positions - prioritize exact matches
            date_candidates = []
            prcp_candidates = []
            
            for j, val in enumerate(row.values):
                val_upper = str(val).strip().upper() if pd.notna(val) else ''
                
                # Score date columns (higher score = better match)
                for date_pattern in ['DATE', 'DAY', 'TIME', 'DT']:
                    if date_pattern in val_upper:
                        score = len(date_pattern) if val_upper == date_pattern else len(date_pattern) - 1
                        date_candidates.append((score, j, val_upper))
                
                # Score precipitation columns (higher score = better match)
                for prcp_pattern in ['PRCP', 'PRECIP', 'PRECIPITATION', 'RAIN', 'PPT']:
                    if prcp_pattern in val_upper:
                        score = len(prcp_pattern) if val_upper == prcp_pattern else len(prcp_pattern) - 1
                        prcp_candidates.append((score, j, val_upper))
            
            # Choose the best matches
            if date_candidates:
                date_candidates.sort(reverse=True)  # Highest score first
                date_col = date_candidates[0][1]
                logger.info(f"Selected DATE column: '{date_candidates[0][2]}' (score: {date_candidates[0][0]})")
            
            if prcp_candidates:
                prcp_candidates.sort(reverse=True)  # Highest score first
                prcp_col = prcp_candidates[0][1]
                logger.info(f"Selected PRCP column: '{prcp_candidates[0][2]}' (score: {prcp_candidates[0][0]})")
            
            break
    
    if header_row is None:
        logger.error("Could not find header row with DATE and PRCP columns.")
        logger.error("Available column headers in first 50 rows:")
        for i, row in raw.iterrows():
            vals = [str(val).strip() for val in row.values if pd.notna(val) and str(val).strip()]
            if vals:  # Only show non-empty rows
                logger.error(f"Row {i}: {vals}")
        raise ValueError("Could not find header row with DATE and PRCP columns.")

    # 3. Load data with correct header
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    
    # 4. Extract the correct columns using their positions
    date_col_name = df.columns[date_col]
    prcp_col_name = df.columns[prcp_col]
    
    logger.info(f"Found DATE column: '{date_col_name}' at position {date_col}")
    logger.info(f"Found PRCP column: '{prcp_col_name}' at position {prcp_col}")
    
    # Select only the date and precipitation columns
    df = df[[date_col_name, prcp_col_name]].copy()
    df.rename(columns={date_col_name: 'DATE', prcp_col_name: 'precip'}, inplace=True)

    # 5. Format data
    logger.info(f"Original DATE column sample: {df['DATE'].head().tolist()}")
    logger.info(f"Original DATE column type: {df['DATE'].dtype}")
    
    # Check if dates are already datetime objects or need parsing
    if pd.api.types.is_datetime64_any_dtype(df['DATE']):
        logger.info("DATE column is already datetime type")
        logger.info(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        logger.info(f"Unique dates: {df['DATE'].nunique()}/{len(df)}")
    else:
        logger.info("Converting DATE column to datetime")
        logger.info(f"Original DATE column sample: {df['DATE'].head().tolist()}")
        logger.info(f"Original DATE column type: {df['DATE'].dtype}")
        
        # Store original date values for format attempts
        original_dates = df['DATE'].copy()
        
        # Convert to datetime - try pandas automatic parsing first
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        logger.info(f"After pd.to_datetime: {df['DATE'].head().tolist()}")
        
        # Check if dates parsed correctly (not all NaT and reasonable years)
        dates_valid = True
        if df['DATE'].isna().all():
            logger.warning("All dates failed to parse with automatic detection")
            dates_valid = False
        elif (df['DATE'].dt.year < 1800).any() or (df['DATE'].dt.year > 2100).any():
            logger.warning("Some dates have unreasonable years - trying explicit formats")
            dates_valid = False
        elif df['DATE'].nunique() < len(df) * 0.1:  # Less than 10% unique dates suggests parsing failure
            logger.warning("Too few unique dates - likely parsing failure")
            dates_valid = False
        
        # If automatic parsing failed, try explicit formats
        if not dates_valid:
            logger.info("Trying explicit date formats...")
            date_formats = [
                '%Y-%m-%d',    # 2000-01-01
                '%m/%d/%Y',    # 1/1/2000
                '%d/%m/%Y',    # 1/1/2000 (day first)
                '%Y/%m/%d',    # 2000/1/1
                '%m-%d-%Y',    # 1-1-2000
                '%d-%m-%Y',    # 1-1-2000 (day first)
                '%Y%m%d',      # 20000101
                '%m/%d/%y',    # 1/1/00
                '%d/%m/%y',    # 1/1/00 (day first)
            ]
            
            for date_format in date_formats:
                try:
                    test_dates = pd.to_datetime(original_dates, format=date_format, errors='coerce')
                    valid_count = test_dates.notna().sum()
                    unique_count = test_dates.nunique()
                    
                    if valid_count > len(df) * 0.8 and unique_count > len(df) * 0.1:  # 80% valid, 10% unique
                        df['DATE'] = test_dates
                        logger.info(f"Successfully parsed {valid_count}/{len(df)} dates using format: {date_format}")
                        logger.info(f"Date range: {test_dates.min()} to {test_dates.max()}")
                        break
                except Exception as e:
                    logger.debug(f"Format {date_format} failed: {e}")
                    continue
            else:
                logger.error("All date format attempts failed")
        else:
            logger.info(f"Date parsing successful. Range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Remove rows with invalid dates
    initial_rows = len(df)
    df = df.dropna(subset=['DATE'])
    final_rows = len(df)
    if initial_rows != final_rows:
        logger.warning(f"Removed {initial_rows - final_rows} rows with invalid dates")
    
    df.set_index('DATE', inplace=True)
    df['precip'] = pd.to_numeric(df['precip'], errors='coerce')
    
    # Handle error codes and negative values
    error_codes = [-99, -999, -9999, 999, 9999, 99999]
    df.loc[df['precip'].isin(error_codes), 'precip'] = float('nan')
    df.loc[df['precip'] < 0, 'precip'] = float('nan')
    
    # Log data quality info
    total_days = len(df)
    valid_precip_days = df['precip'].notna().sum()
    logger.info(f"Loaded {total_days} days, {valid_precip_days} with valid precipitation data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Do NOT fill NaN with 0; leave missing/invalid as NaN

    logger.info(f"Successfully loaded and cleaned data from {file_path}")
    return df
