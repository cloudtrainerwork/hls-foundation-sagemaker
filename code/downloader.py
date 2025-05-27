import json
import morecantile
import os
import rasterio
import requests
import logging
from typing import List, Tuple, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging for Azure ML
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
BASE_URL = "https://d1nzvsko7rbono.cloudfront.net"
BASE_TILE_URL = "{BASE_URL}/mosaic/tiles/{searchid}/WebMercatorQuad/{z}/{x}/{y}.tif"
REGISTER_ENDPOINT = f"{BASE_URL}/mosaic/register"

TILE_URL = {
    "HLSL30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B05&assets=B06&assets=B07",
    "HLSS30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B8A&assets=B11&assets=B12",
}

PROJECTION = "WebMercatorQuad"
TMS = morecantile.tms.get(PROJECTION)
ZOOM_LEVEL = 12

# Azure ML compatible download folder
DOWNLOAD_FOLDER = os.getenv("AZUREML_DATA_DIR", "/tmp/data")


class Downloader:
    """Enhanced downloader class for Azure ML deployment"""
    
    def __init__(self, date: str, layer: str = "HLSL30"):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
        """
        self.layer = layer
        self.date = date
        self.search_id = None
        self.session = self._create_session()
        
        # Ensure download folder exists
        os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
        
        try:
            self.search_id = self.register_new_search()
            logger.info(f"Initialized downloader for {layer} on {date}, search_id: {self.search_id}")
        except Exception as e:
            logger.error(f"Failed to initialize downloader: {str(e)}")
            raise

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout
        session.timeout = 30
        
        return session

    def download_tile(self, x_index: int, y_index: int, filename: str) -> str:
        """
        Download a single tile
        
        Args:
            x_index (int): X tile index
            y_index (int): Y tile index  
            filename (str): Output filename
            
        Returns:
            str: Filename if successful, empty string if failed
        """
        return_filename = filename
        
        try:
            url = TILE_URL[self.layer].format(
                BASE_URL=BASE_URL,
                searchid=self.search_id,
                z=ZOOM_LEVEL,
                x=x_index,
                y=y_index,
            )
            
            logger.debug(f"Downloading tile from: {url}")
            
            response = self.session.get(url)
            
            if response.status_code == 200:
                # Write raw response to file
                with open(filename, "wb") as download_file:
                    download_file.write(response.content)
                
                # Process the raster file
                try:
                    with rasterio.open(filename) as raster_file:
                        profile = raster_file.profile.copy()
                        profile['dtype'] = 'float32'
                        
                        # Create temporary file for processed data
                        temp_filename = f"{filename}.tmp"
                        
                        with rasterio.open(temp_filename, 'w', **profile) as new_file:
                            for band in range(profile['count']):
                                index = band + 1
                                band_data = raster_file.read(index)
                                # Apply scaling factor (convert to reflectance)
                                new_file.write(band_data * 0.0001, index)
                        
                        # Replace original with processed file
                        os.replace(temp_filename, filename)
                        
                except Exception as processing_error:
                    logger.error(f"Error processing raster file {filename}: {str(processing_error)}")
                    # Clean up potentially corrupted file
                    if os.path.exists(filename):
                        os.remove(filename)
                    return_filename = ""
                    
            else:
                logger.warning(f"Failed to download tile {x_index},{y_index}: HTTP {response.status_code}")
                return_filename = ""
                
        except Exception as e:
            logger.error(f"Error downloading tile {x_index},{y_index}: {str(e)}")
            return_filename = ""
            
        return return_filename

    def mkdir(self, foldername: str) -> None:
        """Create directory if it doesn't exist"""
        try:
            os.makedirs(foldername, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {foldername}: {str(e)}")
            raise

    def download_tiles(self, bounding_box: List[float]) -> List[str]:
        """
        Download all tiles for the given bounding box
        
        Args:
            bounding_box (List[float]): [left, bottom, right, top]
            
        Returns:
            List[str]: List of successfully downloaded file paths
        """
        try:
            if not self.search_id:
                raise RuntimeError("No valid search_id available")
                
            logger.info(f"Starting tile download for bounding box: {bounding_box}")
            
            x_tiles, y_tiles = self.tile_indices(bounding_box)
            downloaded_files = []
            total_tiles = (x_tiles[1] - x_tiles[0]) * (y_tiles[1] - y_tiles[0])
            
            logger.info(f"Need to download {total_tiles} tiles")
            
            downloaded_count = 0
            for x_index in range(x_tiles[0], x_tiles[1]):
                for y_index in range(y_tiles[0], y_tiles[1]):
                    # Create layer-specific folder
                    layer_folder = f"{DOWNLOAD_FOLDER}/{self.layer}"
                    self.mkdir(layer_folder)
                    
                    filename = f"{layer_folder}/{self.date}-{x_index}-{y_index}.tif"
                    
                    # Check if file already exists and is valid
                    if os.path.exists(filename) and self._is_valid_raster(filename):
                        downloaded_files.append(filename)
                        downloaded_count += 1
                        logger.debug(f"Using existing tile: {filename}")
                    else:
                        # Download new tile
                        downloaded_file = self.download_tile(x_index, y_index, filename)
                        if downloaded_file:
                            downloaded_files.append(downloaded_file)
                            downloaded_count += 1
                            logger.debug(f"Downloaded new tile: {downloaded_file}")
                        else:
                            logger.warning(f"Failed to download tile {x_index},{y_index}")
            
            logger.info(f"Downloaded {downloaded_count}/{total_tiles} tiles successfully")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error in download_tiles: {str(e)}")
            raise

    def _is_valid_raster(self, filename: str) -> bool:
        """Check if a raster file is valid and readable"""
        try:
            with rasterio.open(filename) as raster:
                # Try to read a small portion to verify file integrity
                raster.read(1, window=rasterio.windows.Window(0, 0, 1, 1))
                return True
        except Exception:
            return False

    def register_new_search(self) -> str:
        """
        Register new search with HLS titiler
        
        Returns:
            str: Search ID for the registered search
        """
        try:
            logger.info(f"Registering new search for {self.layer} on {self.date}")
            
            payload = {
                "datetime": f"{self.date}T00:00:00Z/{self.date}T23:59:59Z",
                "collections": [self.layer],
            }
            
            headers = {
                "Content-Type": "application/json",
                "accept": "application/json"
            }
            
            response = self.session.post(
                REGISTER_ENDPOINT,
                headers=headers,
                data=json.dumps(payload)
            )
            
            response.raise_for_status()  # Raise exception for bad status codes
            
            response_data = response.json()
            search_id = response_data.get("searchid")
            
            if not search_id:
                raise ValueError("No searchid returned from registration")
                
            logger.info(f"Successfully registered search: {search_id}")
            return search_id
            
        except Exception as e:
            logger.error(f"Error registering new search: {str(e)}")
            raise

    def tile_indices(self, bounding_box: List[float]) -> List[List[int]]:
        """
        Extract tile indices based on bounding_box

        Args:
            bounding_box (List[float]): [left, bottom, right, top]

        Returns:
            List[List[int]]: [[start_x, end_x], [start_y, end_y]]
        """
        try:
            # Validate bounding box
            if len(bounding_box) != 4:
                raise ValueError(f"Bounding box must have 4 coordinates, got {len(bounding_box)}")
            
            left, bottom, right, top = bounding_box
            
            # Validate coordinate order
            if left >= right or bottom >= top:
                raise ValueError(f"Invalid bounding box coordinates: {bounding_box}")
            
            # Calculate tile indices
            start_x, start_y, _ = TMS.tile(left, top, ZOOM_LEVEL)
            end_x, end_y, _ = TMS.tile(right, bottom, ZOOM_LEVEL)
            
            # Ensure proper ordering (start <= end)
            start_x, end_x = min(start_x, end_x), max(start_x, end_x) + 1
            start_y, end_y = min(start_y, end_y), max(start_y, end_y) + 1
            
            logger.debug(f"Tile indices for bbox {bounding_box}: x=[{start_x},{end_x}), y=[{start_y},{end_y})")
            
            return [[start_x, end_x], [start_y, end_y]]
            
        except Exception as e:
            logger.error(f"Error calculating tile indices: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.session:
                self.session.close()
            logger.info("Downloader cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
