#!/usr/bin/env python3
"""
Example: Downloading GHCN Data

This example shows how to find and download GHCN Daily (.dly) files
for precipitation analysis using PrecipGen's built-in downloader.
"""

import precipgen as pg
from datetime import datetime

def example_find_stations_by_location():
    """Find stations near a specific location."""
    print("=== Finding Stations by Location ===")
    
    # Example: Find stations near Seattle, WA
    latitude = 47.6062  # Seattle latitude
    longitude = -122.3321  # Seattle longitude
    
    print(f"Searching for stations near Seattle ({latitude}, {longitude})")
    
    # Find nearby stations with at least 20 years of data
    stations = pg.find_nearby_stations(
        latitude=latitude,
        longitude=longitude,
        radius_km=50,  # Within 50 km
        min_years=20   # At least 20 years of data
    )
    
    print(f"\nFound {len(stations)} stations:")
    for i, station in enumerate(stations[:5]):  # Show first 5
        print(f"{i+1}. {station['id']}: {station['name']}")
        print(f"   Distance: {station['distance_km']:.1f} km")
        print(f"   Data: {station['first_year']}-{station['last_year']} ({station['years_available']} years)")
        print(f"   Location: {station['latitude']:.4f}, {station['longitude']:.4f}")
        print()
    
    return stations

def example_find_stations_by_name():
    """Find stations by name pattern."""
    print("=== Finding Stations by Name ===")
    
    # Create downloader instance
    downloader = pg.GHCNDownloader()
    
    # Search for stations with "Seattle" in the name
    stations = downloader.find_stations_by_name("Seattle", min_years=15)
    
    print(f"Found {len(stations)} stations with 'Seattle' in name:")
    for station in stations:
        print(f"- {station['id']}: {station['name']}")
        print(f"  Data: {station['first_year']}-{station['last_year']} ({station['years_available']} years)")
    
    return stations

def example_download_and_analyze():
    """Download station data and perform analysis."""
    print("=== Download and Analyze ===")
    
    # Use Seattle-Tacoma International Airport station
    station_id = "USW00024233"  # Seattle-Tacoma Intl Airport
    
    print(f"Downloading data for station {station_id}...")
    
    # Download the .dly file
    try:
        file_path = pg.download_station(station_id)
        print(f"Downloaded to: {file_path}")
        
        # Parse the downloaded file
        parser = pg.GHCNParser(file_path)
        ghcn_data = parser.parse_dly_file(file_path)
        precip_data = parser.extract_precipitation(ghcn_data)
        
        print(f"\nData summary:")
        print(f"- Total records: {len(precip_data)}")
        print(f"- Date range: {precip_data.index.min()} to {precip_data.index.max()}")
        print(f"- Non-zero precipitation days: {(precip_data > 0).sum()}")
        print(f"- Mean daily precipitation: {precip_data.mean():.2f} mm")
        print(f"- Max daily precipitation: {precip_data.max():.2f} mm")
        
        # Validate data quality
        validator = pg.DataValidator(pg.QualityConfig())
        quality_report = validator.assess_data_quality(precip_data, site_id=station_id)
        
        print(f"\nData quality:")
        print(f"- Completeness: {quality_report.completeness_percentage:.1f}%")
        print(f"- Acceptable for analysis: {quality_report.is_acceptable}")
        
        if quality_report.is_acceptable:
            # Perform parameter estimation
            print(f"\nPerforming parameter estimation...")
            engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
            manifest = engine.generate_parameter_manifest()
            
            print("✓ Parameter estimation completed")
            
            # Generate some synthetic data
            sim = pg.SimulationEngine(manifest, random_seed=42)
            sim.initialize(datetime(2025, 1, 1))
            synthetic = [sim.step() for _ in range(365)]
            
            print(f"✓ Generated {len(synthetic)} days of synthetic precipitation")
            print(f"  Synthetic annual total: {sum(synthetic):.1f} mm")
            print(f"  Historical annual mean: {precip_data.groupby(precip_data.index.year).sum().mean():.1f} mm")
        
        return file_path, precip_data
        
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return None, None

def example_batch_download():
    """Download multiple stations at once."""
    print("=== Batch Download ===")
    
    # List of Pacific Northwest stations
    station_ids = [
        "USW00024233",  # Seattle-Tacoma Intl Airport
        "USW00024229",  # Boeing Field
        "USC00455622",  # Portland Intl Airport
        "USC00356750",  # Spokane Intl Airport
    ]
    
    downloader = pg.GHCNDownloader()
    
    print(f"Downloading {len(station_ids)} stations...")
    
    # Download with 2-second delay between requests (be nice to NOAA servers)
    files = downloader.download_multiple_stations(station_ids, delay_seconds=2.0)
    
    print(f"\nSuccessfully downloaded {len(files)} files:")
    for station_id, file_path in files.items():
        # Get station info
        info = downloader.get_station_info(station_id)
        if info:
            print(f"- {station_id}: {info['name']}")
            print(f"  File: {file_path}")
            print(f"  Data: {info.get('first_year', 'N/A')}-{info.get('last_year', 'N/A')}")
    
    return files

def main():
    """Run all examples."""
    print("GHCN Data Download Examples")
    print("=" * 50)
    
    try:
        # Example 1: Find stations by location
        stations_by_location = example_find_stations_by_location()
        
        print("\n" + "=" * 50)
        
        # Example 2: Find stations by name
        stations_by_name = example_find_stations_by_name()
        
        print("\n" + "=" * 50)
        
        # Example 3: Download and analyze single station
        file_path, data = example_download_and_analyze()
        
        print("\n" + "=" * 50)
        
        # Example 4: Batch download
        batch_files = example_batch_download()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
        # Summary
        print(f"\nSummary:")
        print(f"- Found {len(stations_by_location)} stations near Seattle")
        print(f"- Found {len(stations_by_name)} stations with 'Seattle' in name")
        if file_path:
            print(f"- Downloaded and analyzed: {file_path}")
        print(f"- Batch downloaded {len(batch_files)} station files")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()