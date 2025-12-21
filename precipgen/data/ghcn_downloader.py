"""
GHCN data downloader for PrecipGen library.

This module provides utilities to download GHCN Daily (.dly) files
from NOAA's servers and find nearby weather stations.
"""

import requests
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from io import StringIO
import time
from ..utils.logging_config import get_logger
from ..utils.exceptions import DataLoadError, NetworkError


class GHCNDownloader:
    """
    Download GHCN Daily data files from NOAA servers.
    
    Provides methods to search for stations, download .dly files,
    and manage local GHCN data collections.
    """
    
    BASE_URL = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/all/"
    STATIONS_URL = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
    INVENTORY_URL = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize GHCN downloader.
        
        Args:
            cache_dir: Directory to cache downloaded files (default: ./ghcn_data)
        """
        self.logger = get_logger('ghcn_downloader')
        self.cache_dir = Path(cache_dir or "ghcn_data")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache for station metadata
        self._stations_cache = None
        self._inventory_cache = None
    
    def find_stations_by_location(self, latitude: float, longitude: float, 
                                 radius_km: float = 50, 
                                 min_years: int = 10) -> List[Dict]:
        """
        Find GHCN stations near a geographic location.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees  
            radius_km: Search radius in kilometers
            min_years: Minimum years of precipitation data required
            
        Returns:
            List of station dictionaries with metadata
            
        Example:
            >>> downloader = GHCNDownloader()
            >>> stations = downloader.find_stations_by_location(40.7128, -74.0060, radius_km=25)
            >>> for station in stations[:3]:
            ...     print(f"{station['id']}: {station['name']} ({station['distance_km']:.1f} km)")
        """
        self.logger.info(f"Searching for stations near {latitude:.4f}, {longitude:.4f}")
        
        # Load station metadata
        stations_df = self._get_stations_metadata()
        inventory_df = self._get_inventory_metadata()
        
        # Calculate distances using Haversine formula
        stations_df['distance_km'] = self._calculate_distance(
            latitude, longitude, 
            stations_df['latitude'], stations_df['longitude']
        )
        
        # Filter by distance
        nearby_stations = stations_df[stations_df['distance_km'] <= radius_km].copy()
        
        if len(nearby_stations) == 0:
            self.logger.warning(f"No stations found within {radius_km} km")
            return []
        
        # Add precipitation data availability
        precip_inventory = inventory_df[inventory_df['element'] == 'PRCP'].copy()
        
        results = []
        for _, station in nearby_stations.iterrows():
            station_id = station['id']
            station_precip = precip_inventory[precip_inventory['id'] == station_id]
            
            if len(station_precip) > 0:
                # Calculate years of data
                first_year = station_precip['firstyear'].min()
                last_year = station_precip['lastyear'].max()
                years_available = last_year - first_year + 1
                
                if years_available >= min_years:
                    results.append({
                        'id': station_id,
                        'name': station['name'],
                        'latitude': station['latitude'],
                        'longitude': station['longitude'],
                        'elevation_m': station['elevation'],
                        'distance_km': station['distance_km'],
                        'first_year': first_year,
                        'last_year': last_year,
                        'years_available': years_available
                    })
        
        # Sort by distance
        results.sort(key=lambda x: x['distance_km'])
        
        self.logger.info(f"Found {len(results)} stations with {min_years}+ years of precipitation data")
        return results
    
    def find_stations_by_name(self, name_pattern: str, 
                             min_years: int = 10) -> List[Dict]:
        """
        Find GHCN stations by name pattern.
        
        Args:
            name_pattern: Station name pattern (case-insensitive)
            min_years: Minimum years of precipitation data required
            
        Returns:
            List of station dictionaries with metadata
            
        Example:
            >>> downloader = GHCNDownloader()
            >>> stations = downloader.find_stations_by_name("Seattle")
            >>> for station in stations:
            ...     print(f"{station['id']}: {station['name']}")
        """
        self.logger.info(f"Searching for stations matching '{name_pattern}'")
        
        stations_df = self._get_stations_metadata()
        inventory_df = self._get_inventory_metadata()
        
        # Filter by name (case-insensitive)
        matching_stations = stations_df[
            stations_df['name'].str.contains(name_pattern, case=False, na=False)
        ].copy()
        
        if len(matching_stations) == 0:
            self.logger.warning(f"No stations found matching '{name_pattern}'")
            return []
        
        # Add precipitation data availability
        precip_inventory = inventory_df[inventory_df['element'] == 'PRCP'].copy()
        
        results = []
        for _, station in matching_stations.iterrows():
            station_id = station['id']
            station_precip = precip_inventory[precip_inventory['id'] == station_id]
            
            if len(station_precip) > 0:
                first_year = station_precip['firstyear'].min()
                last_year = station_precip['lastyear'].max()
                years_available = last_year - first_year + 1
                
                if years_available >= min_years:
                    results.append({
                        'id': station_id,
                        'name': station['name'],
                        'latitude': station['latitude'],
                        'longitude': station['longitude'],
                        'elevation_m': station['elevation'],
                        'first_year': first_year,
                        'last_year': last_year,
                        'years_available': years_available
                    })
        
        self.logger.info(f"Found {len(results)} matching stations with {min_years}+ years of data")
        return results
    
    def download_station_data(self, station_id: str, 
                             force_refresh: bool = False) -> str:
        """
        Download GHCN .dly file for a specific station.
        
        Args:
            station_id: GHCN station identifier (e.g., 'USC00123456')
            force_refresh: Force re-download even if file exists
            
        Returns:
            Path to downloaded .dly file
            
        Raises:
            NetworkError: If download fails
            DataLoadError: If file cannot be saved
            
        Example:
            >>> downloader = GHCNDownloader()
            >>> file_path = downloader.download_station_data('USC00305426')
            >>> print(f"Downloaded: {file_path}")
        """
        file_path = self.cache_dir / f"{station_id}.dly"
        
        # Check if file already exists
        if file_path.exists() and not force_refresh:
            self.logger.info(f"Using cached file: {file_path}")
            return str(file_path)
        
        # Download from NOAA
        url = f"{self.BASE_URL}{station_id}.dly"
        self.logger.info(f"Downloading {station_id} from {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            self.logger.info(f"Downloaded {len(response.text)} bytes to {file_path}")
            return str(file_path)
            
        except requests.RequestException as e:
            error_msg = f"Failed to download {station_id}: {e}"
            self.logger.error(error_msg)
            raise NetworkError(error_msg) from e
        except IOError as e:
            error_msg = f"Failed to save {station_id}: {e}"
            self.logger.error(error_msg)
            raise DataLoadError(error_msg) from e
    
    def download_multiple_stations(self, station_ids: List[str], 
                                  delay_seconds: float = 1.0) -> Dict[str, str]:
        """
        Download multiple GHCN .dly files with rate limiting.
        
        Args:
            station_ids: List of GHCN station identifiers
            delay_seconds: Delay between downloads to be respectful to NOAA servers
            
        Returns:
            Dictionary mapping station_id -> file_path for successful downloads
            
        Example:
            >>> downloader = GHCNDownloader()
            >>> stations = ['USC00305426', 'USC00305428']
            >>> files = downloader.download_multiple_stations(stations)
            >>> print(f"Downloaded {len(files)} files")
        """
        self.logger.info(f"Downloading {len(station_ids)} station files")
        
        results = {}
        failed = []
        
        for i, station_id in enumerate(station_ids):
            try:
                file_path = self.download_station_data(station_id)
                results[station_id] = file_path
                self.logger.info(f"Progress: {i+1}/{len(station_ids)} - {station_id} ✓")
                
                # Rate limiting
                if i < len(station_ids) - 1:  # Don't delay after last download
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                self.logger.error(f"Failed to download {station_id}: {e}")
                failed.append(station_id)
        
        self.logger.info(f"Downloaded {len(results)} files successfully")
        if failed:
            self.logger.warning(f"Failed downloads: {failed}")
        
        return results
    
    def get_station_info(self, station_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific station.
        
        Args:
            station_id: GHCN station identifier
            
        Returns:
            Station information dictionary or None if not found
        """
        stations_df = self._get_stations_metadata()
        inventory_df = self._get_inventory_metadata()
        
        station_info = stations_df[stations_df['id'] == station_id]
        if len(station_info) == 0:
            return None
        
        station = station_info.iloc[0]
        
        # Get precipitation data availability
        station_inventory = inventory_df[
            (inventory_df['id'] == station_id) & 
            (inventory_df['element'] == 'PRCP')
        ]
        
        result = {
            'id': station['id'],
            'name': station['name'],
            'latitude': station['latitude'],
            'longitude': station['longitude'],
            'elevation_m': station['elevation']
        }
        
        if len(station_inventory) > 0:
            result.update({
                'first_year': station_inventory['firstyear'].min(),
                'last_year': station_inventory['lastyear'].max(),
                'years_available': station_inventory['lastyear'].max() - station_inventory['firstyear'].min() + 1
            })
        
        return result
    
    def _get_stations_metadata(self) -> pd.DataFrame:
        """Load and cache GHCN stations metadata."""
        if self._stations_cache is None:
            self.logger.info("Loading GHCN stations metadata")
            
            cache_file = self.cache_dir / "ghcnd-stations.txt"
            
            # Try to load from cache first
            if cache_file.exists():
                self.logger.info("Using cached stations metadata")
                self._stations_cache = self._parse_stations_file(str(cache_file))
            else:
                # Download from NOAA
                self.logger.info("Downloading stations metadata from NOAA")
                response = requests.get(self.STATIONS_URL, timeout=60)
                response.raise_for_status()
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                self._stations_cache = self._parse_stations_file(str(cache_file))
        
        return self._stations_cache
    
    def _get_inventory_metadata(self) -> pd.DataFrame:
        """Load and cache GHCN inventory metadata."""
        if self._inventory_cache is None:
            self.logger.info("Loading GHCN inventory metadata")
            
            cache_file = self.cache_dir / "ghcnd-inventory.txt"
            
            # Try to load from cache first
            if cache_file.exists():
                self.logger.info("Using cached inventory metadata")
                self._inventory_cache = self._parse_inventory_file(str(cache_file))
            else:
                # Download from NOAA
                self.logger.info("Downloading inventory metadata from NOAA")
                response = requests.get(self.INVENTORY_URL, timeout=60)
                response.raise_for_status()
                
                # Save to cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                self._inventory_cache = self._parse_inventory_file(str(cache_file))
        
        return self._inventory_cache
    
    def _parse_stations_file(self, file_path: str) -> pd.DataFrame:
        """Parse GHCN stations file format."""
        # GHCN stations file format:
        # ID: positions 1-11
        # LATITUDE: positions 13-20
        # LONGITUDE: positions 22-30
        # ELEVATION: positions 32-37
        # NAME: positions 42-71
        
        stations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) >= 71:
                    stations.append({
                        'id': line[0:11].strip(),
                        'latitude': float(line[12:20].strip()),
                        'longitude': float(line[21:30].strip()),
                        'elevation': float(line[31:37].strip()),
                        'name': line[41:71].strip()
                    })
        
        return pd.DataFrame(stations)
    
    def _parse_inventory_file(self, file_path: str) -> pd.DataFrame:
        """Parse GHCN inventory file format."""
        # GHCN inventory file format:
        # ID: positions 1-11
        # ELEMENT: positions 32-35
        # FIRSTYEAR: positions 37-40
        # LASTYEAR: positions 42-45
        
        inventory = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) >= 45:
                    inventory.append({
                        'id': line[0:11].strip(),
                        'element': line[31:35].strip(),
                        'firstyear': int(line[36:40].strip()),
                        'lastyear': int(line[41:45].strip())
                    })
        
        return pd.DataFrame(inventory)
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2_series: pd.Series, lon2_series: pd.Series) -> pd.Series:
        """Calculate distance using Haversine formula."""
        import numpy as np
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2_series)
        lon2_rad = np.radians(lon2_series)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371.0
        
        return r * c


def find_nearby_stations(latitude: float, longitude: float, 
                        radius_km: float = 50, min_years: int = 10) -> List[Dict]:
    """
    Convenience function to find nearby GHCN stations.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        radius_km: Search radius in kilometers
        min_years: Minimum years of precipitation data
        
    Returns:
        List of station dictionaries
        
    Example:
        >>> stations = find_nearby_stations(40.7128, -74.0060)  # NYC
        >>> print(f"Found {len(stations)} stations near NYC")
    """
    downloader = GHCNDownloader()
    return downloader.find_stations_by_location(latitude, longitude, radius_km, min_years)


def download_station(station_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Convenience function to download a single GHCN station file.
    
    Args:
        station_id: GHCN station identifier
        cache_dir: Directory to save file (default: ./ghcn_data)
        
    Returns:
        Path to downloaded .dly file
        
    Example:
        >>> file_path = download_station('USC00305426')
        >>> print(f"Downloaded to: {file_path}")
    """
    downloader = GHCNDownloader(cache_dir)
    return downloader.download_station_data(station_id)