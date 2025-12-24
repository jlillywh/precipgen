"""
Simplified Quality Assessment Example

Shows how to use the improved quality assessment API with presets
and automatic fallback for easier data validation.
"""

import precipgen as pg

def main():
    # Download and parse data (same as before)
    station_id = 'USW00024127'  # Salt Lake City
    downloader = pg.GHCNDownloader(cache_dir='data')
    dly_path = downloader.download_station_data(station_id)
    
    parser = pg.GHCNParser(dly_path)
    ghcn_data = parser.parse_dly_file(dly_path)
    precip_data = parser.extract_precipitation(ghcn_data)
    
    print("=== OPTION 1: Automatic Fallback (Recommended) ===")
    # This will try standard -> lenient -> permissive until data passes
    validator = pg.DataValidator()
    quality_report = validator.assess_with_fallback(
        precip_data, 
        quality_flags=ghcn_data['quality_flag'],
        site_id=station_id
    )
    
    print(f"Acceptable: {quality_report.is_acceptable}")
    print("Recommendations:")
    for rec in quality_report.recommendations:
        print(f"  - {rec}")
    
    print("\n=== OPTION 2: Use Specific Quality Preset ===")
    # Use a lenient preset directly
    lenient_validator = pg.DataValidator(pg.QualityPresets.lenient())
    lenient_report = lenient_validator.assess_data_quality(
        precip_data,
        quality_flags=ghcn_data['quality_flag'],
        site_id=station_id
    )
    
    print(f"Lenient assessment - Acceptable: {lenient_report.is_acceptable}")
    
    print("\n=== OPTION 3: List Available Presets ===")
    presets = pg.QualityPresets.list_presets()
    print("Available quality presets:")
    for name, description in presets.items():
        print(f"  {name}: {description}")
    
    print("\n=== OPTION 4: Custom Quality Levels ===")
    # Try only specific quality levels
    custom_report = validator.assess_with_fallback(
        precip_data,
        quality_flags=ghcn_data['quality_flag'],
        site_id=station_id,
        quality_levels=['lenient', 'permissive']  # Skip standard
    )
    
    print(f"Custom levels assessment - Acceptable: {custom_report.is_acceptable}")

if __name__ == "__main__":
    main()