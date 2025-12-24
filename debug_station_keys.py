#!/usr/bin/env python3
"""
Debug what keys are returned by find_nearby_stations.
"""

import precipgen as pg

def main():
    # Sea-Tac Airport coordinates
    seatac_lat = 47.4502
    seatac_lon = -122.3088
    
    print("=== Debugging station data structure ===")
    
    # Find just one station to see the structure
    stations = pg.find_nearby_stations(
        latitude=seatac_lat,
        longitude=seatac_lon,
        radius_km=50,
        min_years=10
    )
    
    if stations:
        print(f"Found {len(stations)} stations")
        print("First station keys:", list(stations[0].keys()))
        print("First station data:")
        for key, value in stations[0].items():
            print(f"  {key}: {value}")
    else:
        print("No stations found")

if __name__ == "__main__":
    main()