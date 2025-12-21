#!/usr/bin/env python3
"""
Simple GHCN Download Example

This example shows how to download a specific GHCN station file
when you already know the station ID.
"""

import precipgen as pg
from datetime import datetime

def download_known_station():
    """Download data for a known station ID."""
    
    # Seattle-Tacoma International Airport
    station_id = "USW00024233"
    
    print(f"Downloading GHCN data for station {station_id}...")
    print("(This may take a moment on first download)")
    
    try:
        # Download the .dly file
        file_path = pg.download_station(station_id)
        print(f"✓ Downloaded to: {file_path}")
        
        # Parse the downloaded file
        print("Parsing GHCN data...")
        parser = pg.GHCNParser(file_path)
        ghcn_data = parser.parse_dly_file(file_path)
        precip_data = parser.extract_precipitation(ghcn_data)
        
        print(f"✓ Parsed {len(precip_data)} daily records")
        print(f"  Date range: {precip_data.index.min().date()} to {precip_data.index.max().date()}")
        print(f"  Non-zero days: {(precip_data > 0).sum()}")
        print(f"  Mean daily: {precip_data.mean():.2f} mm")
        
        # Validate data quality
        print("Checking data quality...")
        validator = pg.DataValidator(pg.QualityConfig())
        quality_report = validator.assess_data_quality(precip_data, site_id=station_id)
        
        print(f"✓ Data completeness: {quality_report.completeness_percentage:.1f}%")
        print(f"✓ Acceptable for analysis: {quality_report.is_acceptable}")
        
        if quality_report.is_acceptable:
            # Perform analysis
            print("Analyzing precipitation patterns...")
            engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
            manifest = engine.generate_parameter_manifest()
            
            print("✓ Parameter estimation completed")
            
            # Generate synthetic data
            print("Generating synthetic precipitation...")
            sim = pg.SimulationEngine(manifest, random_seed=42)
            sim.initialize(datetime(2025, 1, 1))
            synthetic = [sim.step() for _ in range(365)]
            
            print(f"✓ Generated {len(synthetic)} days of synthetic data")
            
            # Compare statistics
            historical_annual = precip_data.groupby(precip_data.index.year).sum()
            print(f"\nComparison:")
            print(f"  Historical annual mean: {historical_annual.mean():.1f} mm")
            print(f"  Historical annual std:  {historical_annual.std():.1f} mm")
            print(f"  Synthetic annual total: {sum(synthetic):.1f} mm")
            
            return file_path, precip_data, synthetic
        else:
            print("❌ Data quality insufficient for analysis")
            print("Issues found:")
            for issue in quality_report.issues:
                print(f"  - {issue}")
            return file_path, precip_data, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def main():
    """Run the simple download example."""
    print("Simple GHCN Download Example")
    print("=" * 40)
    
    file_path, data, synthetic = download_known_station()
    
    if file_path:
        print(f"\n✅ Success! GHCN data downloaded and processed.")
        print(f"File saved to: {file_path}")
        
        if synthetic:
            print("Ready for further analysis or integration into your models!")
        else:
            print("Data downloaded but quality issues prevent analysis.")
    else:
        print("\n❌ Download failed. Check your internet connection.")
    
    print("\nNext steps:")
    print("1. Try the full download example: docs/examples/download_ghcn_data.py")
    print("2. Read the GHCN guide: docs/getting_ghcn_data.md")
    print("3. Explore other examples in docs/examples/")

if __name__ == "__main__":
    main()