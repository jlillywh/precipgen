#!/usr/bin/env python
"""
Precipitation Parameter Analyzer CLI
Usage:
    python precipgen_params.py --input_file "path_to_file.xlsx"
    python precipgen_params.py --input_file "path_to_file.xlsx" --sheet_name "Original"
    python precipgen_params.py --input_file "path_to_file.xlsx" --sheet_name "Original" --config "custom_config.yaml"

Arguments:
    --input_file: Path to the Excel file containing daily precipitation data
    --sheet_name: Name of the Excel sheet to read (optional, uses first sheet if not specified)
    --config: Path to configuration file (optional, default: config.yaml)

Configuration:
    Edit config.yaml to adjust:
    - wet_day_threshold: Minimum precipitation (inches) to be considered a wet day
    - sliding_window_years: Number of years for sliding window analysis

Outputs a results Excel file in the same folder as the input, prefixed with 'results_'.
"""
import argparse
import os
import sys
import yaml
from src.data_loader import load_precipitation_data
from src.calculations import calculate_monthly_parameters
from src.analysis import analyze_parameter_trends
from src.output_generator import write_results_to_excel

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    default_config = {
        'wet_day_threshold': 0.001,
        'sliding_window_years': 30
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
    
    return default_config

def main():
    parser = argparse.ArgumentParser(description="Precipitation Parameter Analyzer")
    parser.add_argument("--input_file", required=True, help="Path to the input Excel file")
    parser.add_argument("--sheet_name", default=None, help="Name of the Excel sheet to read (default: first sheet)")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file (default: config.yaml)")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    wet_day_threshold = config['wet_day_threshold']
    sliding_window_years = config['sliding_window_years']

    input_path = args.input_file
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file '{input_path}' does not exist.")
        sys.exit(1)

    dir_name, file_name = os.path.split(input_path)
    output_file = os.path.join(dir_name, f"results_{file_name}")

    print(f"Configuration loaded:")
    print(f"  Wet day threshold: {wet_day_threshold} inches")
    print(f"  Sliding window years: {sliding_window_years}")
    print()
    sheet_info = f" (sheet: {args.sheet_name})" if args.sheet_name else ""
    print(f"Loading data from {input_path}{sheet_info}")
    data = load_precipitation_data(input_path, sheet_name=args.sheet_name)
    print(f"Calculating monthly parameters with wet day threshold: {wet_day_threshold} inches...")
    monthly_params = calculate_monthly_parameters(data, wet_day_threshold=wet_day_threshold)
    print("Performing sliding window analysis...")
    annual_analysis = analyze_parameter_trends(data, window_years=sliding_window_years)
    print(f"Writing results to {output_file}")
    write_results_to_excel(monthly_params, annual_analysis, output_file, raw_df=data)
    print(f"Analysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main()
