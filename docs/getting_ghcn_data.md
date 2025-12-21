# Getting GHCN Data - Easy Guide

PrecipGen includes built-in tools to find and download GHCN Daily (.dly) files automatically. No need to manually browse NOAA websites!

## Quick Start

### 1. Find Stations Near Your Location

```python
import precipgen as pg

# Find stations near New York City
stations = pg.find_nearby_stations(
    latitude=40.7128,   # NYC latitude
    longitude=-74.0060, # NYC longitude
    radius_km=50,       # Within 50 km
    min_years=20        # At least 20 years of data
)

# Show the best options
for i, station in enumerate(stations[:3]):
    print(f"{i+1}. {station['id']}: {station['name']}")
    print(f"   Distance: {station['distance_km']:.1f} km")
    print(f"   Data: {station['first_year']}-{station['last_year']}")
```

### 2. Download Station Data

```python
# Download data for the closest station
station_id = stations[0]['id']  # Use the first (closest) station
file_path = pg.download_station(station_id)

print(f"Downloaded to: {file_path}")
```

### 3. Use the Downloaded Data

```python
# Parse and analyze the downloaded file
parser = pg.GHCNParser(file_path)
ghcn_data = parser.parse_dly_file(file_path)
precip_data = parser.extract_precipitation(ghcn_data)

# Now you can use it with PrecipGen
engine = pg.AnalyticalEngine(precip_data)
manifest = engine.generate_parameter_manifest()
```

## Complete Example

Here's a complete workflow from finding stations to generating synthetic data:

```python
import precipgen as pg
from datetime import datetime

# 1. Find stations near your location
print("Finding stations near Seattle...")
stations = pg.find_nearby_stations(47.6062, -122.3321, radius_km=25)

if len(stations) == 0:
    print("No stations found! Try increasing the radius.")
    exit()

# 2. Download the best station
best_station = stations[0]
print(f"Using: {best_station['name']} ({best_station['distance_km']:.1f} km away)")

file_path = pg.download_station(best_station['id'])
print(f"Downloaded: {file_path}")

# 3. Load and validate the data
parser = pg.GHCNParser(file_path)
ghcn_data = parser.parse_dly_file(file_path)
precip_data = parser.extract_precipitation(ghcn_data)

validator = pg.DataValidator(pg.QualityConfig())
quality_report = validator.assess_data_quality(precip_data)

print(f"Data quality: {quality_report.completeness_percentage:.1f}% complete")

if not quality_report.is_acceptable:
    print("Data quality insufficient. Try a different station.")
    exit()

# 4. Analyze and generate synthetic data
print("Analyzing historical data...")
engine = pg.AnalyticalEngine(precip_data, wet_day_threshold=0.001)
manifest = engine.generate_parameter_manifest()

print("Generating synthetic precipitation...")
sim = pg.SimulationEngine(manifest, random_seed=42)
sim.initialize(datetime(2025, 1, 1))
synthetic = [sim.step() for _ in range(365)]

print(f"✓ Generated {len(synthetic)} days of synthetic data")
print(f"Historical annual mean: {precip_data.groupby(precip_data.index.year).sum().mean():.1f} mm")
print(f"Synthetic annual total: {sum(synthetic):.1f} mm")
```

## Advanced Usage

### Find Stations by Name

```python
downloader = pg.GHCNDownloader()

# Find all stations with "Seattle" in the name
seattle_stations = downloader.find_stations_by_name("Seattle", min_years=15)

for station in seattle_stations:
    print(f"{station['id']}: {station['name']}")
```

### Download Multiple Stations

```python
station_ids = ["USW00024233", "USW00024229", "USC00455622"]

downloader = pg.GHCNDownloader()
files = downloader.download_multiple_stations(
    station_ids, 
    delay_seconds=2.0  # Be nice to NOAA servers
)

print(f"Downloaded {len(files)} files")
```

### Get Station Information

```python
downloader = pg.GHCNDownloader()
info = downloader.get_station_info("USW00024233")

if info:
    print(f"Station: {info['name']}")
    print(f"Location: {info['latitude']}, {info['longitude']}")
    print(f"Elevation: {info['elevation_m']} m")
    print(f"Data: {info['first_year']}-{info['last_year']}")
```

## Common Locations

Here are coordinates for some major cities to get you started:

| City | Latitude | Longitude |
|------|----------|-----------|
| New York City | 40.7128 | -74.0060 |
| Los Angeles | 34.0522 | -118.2437 |
| Chicago | 41.8781 | -87.6298 |
| Houston | 29.7604 | -95.3698 |
| Phoenix | 33.4484 | -112.0740 |
| Philadelphia | 39.9526 | -75.1652 |
| San Antonio | 29.4241 | -98.4936 |
| San Diego | 32.7157 | -117.1611 |
| Dallas | 32.7767 | -96.7970 |
| San Jose | 37.3382 | -121.8863 |
| Seattle | 47.6062 | -122.3321 |
| Denver | 39.7392 | -104.9903 |
| Boston | 42.3601 | -71.0589 |
| Miami | 25.7617 | -80.1918 |
| Atlanta | 33.7490 | -84.3880 |

## Tips

1. **Start with a small radius** (25-50 km) and increase if no stations are found
2. **Require at least 15-20 years** of data for reliable parameter estimation
3. **Check data quality** before proceeding with analysis
4. **Be patient** with downloads - NOAA servers can be slow
5. **Files are cached** - subsequent downloads of the same station are instant

## Troubleshooting

### No stations found
- Increase the search radius
- Reduce the minimum years requirement
- Check your coordinates (latitude/longitude)

### Download fails
- Check your internet connection
- NOAA servers may be temporarily unavailable
- Try again later

### Data quality issues
- Try a different station
- Check if the station has recent data
- Some stations may have gaps in their records

## What's Next?

Once you have your .dly file:

1. **Validate the data** using `DataValidator`
2. **Analyze parameters** with `AnalyticalEngine`
3. **Generate synthetic data** with `SimulationEngine`
4. **Check out the examples** in `docs/examples/`

The built-in downloader makes it easy to get started with real climate data for any location!