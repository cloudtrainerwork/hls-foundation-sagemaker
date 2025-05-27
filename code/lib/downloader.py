import json
import morecantile
import os
import rasterio
import requests
import logging
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azureml.core import Run
from typing import Optional, List

BASE_URL = "https://d1nzvsko7rbono.cloudfront.net"
BASE_TILE_URL = "{BASE_URL}/mosaic/tiles/{searchid}/WebMercatorQuad/{z}/{x}/{y}.tif"

REGISTER_ENDPOINT = f"{BASE_URL}/mosaic/register"

TILE_URL = {
    "HLSL30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B05&assets=B06&assets=B07&&Fmask",
    "HLSS30": f"{BASE_TILE_URL}?assets=B02&assets=B03&assets=B04&assets=B8A&assets=B11&assets=B12&&Fmask",
}

PROJECTION = "WebMercatorQuad"
TMS = morecantile.tms.get(PROJECTION)
ZOOM_LEVEL = 12
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../data")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Downloader:
    def __init__(self, date: str, layer: str = "HLSL30", 
                 azure_storage_account: Optional[str] = None,
                 azure_container: Optional[str] = None,
                 use_azure_storage: bool = False):
        """
        Initialize Downloader
        Args:
            date (str): Date in the format of 'yyyy-mm-dd'
            layer (str): any of HLSL30, HLSS30
            azure_storage_account (str): Azure Storage Account name (optional)
            azure_container (str): Azure Blob Container name (optional)
            use_azure_storage (bool): Whether to upload files to Azure Storage
        """
        self.layer = layer
        self.date = date
        self.use_azure_storage = use_azure_storage
        self.azure_storage_account = azure_storage_account
        self.azure_container = azure_container
        
        # Initialize Azure ML run context if available
        try:
            self.run = Run.get_context()
            logger.info("Azure ML run context initialized")
        except Exception:
            self.run = None
            logger.info("Not running in Azure ML context")
        
        # Initialize Azure Storage client if requested
        if self.use_azure_storage and azure_storage_account:
            try:
                credential = DefaultAzureCredential()
                storage_url = f"https://{azure_storage_account}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=storage_url, 
                    credential=credential
                )
                logger.info(f"Azure Storage client initialized for {azure_storage_account}")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Storage: {e}")
                self.blob_service_client = None
        else:
            self.blob_service_client = None
        
        self.search_id = self.register_new_search()

    def download_tile(self, x_index: int, y_index: int, filename: str) -> str:
        """Download a single tile and optionally upload to Azure Storage"""
        return_filename = filename
        
        try:
            response = requests.get(
                TILE_URL[self.layer].format(
                    BASE_URL=BASE_URL,
                    searchid=self.search_id,
                    z=ZOOM_LEVEL,
                    x=x_index,
                    y=y_index,
                ),
                timeout=30
            )
            
            if response.status_code == 200:
                # Save file locally
                with open(filename, "wb") as download_file:
                    download_file.write(response.content)
                
                # Process the raster file
                raster_file = rasterio.open(filename)
                profile = raster_file.profile
                profile['dtype'] = 'float32'
                
                with rasterio.open(filename, 'w', **profile) as new_file:
                    for band in range(profile['count']):
                        index = band + 1
                        new_file.write(raster_file.read(index) * 0.0001, index)
                raster_file.close()
                
                # Upload to Azure Storage if configured
                if self.use_azure_storage and self.blob_service_client and self.azure_container:
                    self._upload_to_azure_storage(filename, x_index, y_index)
                
                # Log to Azure ML if available
                if self.run:
                    self.run.log("tiles_downloaded", 1)
                
                logger.info(f"Successfully downloaded tile {x_index}-{y_index}")
                
            else:
                logger.warning(f"Failed to download tile {x_index}-{y_index}: {response.status_code}")
                return_filename = ""
                
        except Exception as e:
            logger.error(f"Error downloading tile {x_index}-{y_index}: {e}")
            return_filename = ""
            
        return return_filename

    def _upload_to_azure_storage(self, filename: str, x_index: int, y_index: int):
        """Upload file to Azure Blob Storage"""
        try:
            blob_name = f"{self.layer}/{self.date}/{os.path.basename(filename)}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.azure_container, 
                blob=blob_name
            )
            
            with open(filename, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Uploaded {filename} to Azure Storage as {blob_name}")
            
        except Exception as e:
            logger.error(f"Failed to upload {filename} to Azure Storage: {e}")

    def mkdir(self, foldername: str):
        """Create directory if it doesn't exist"""
        if not os.path.exists(foldername):
            os.makedirs(foldername)
            logger.info(f"Created directory: {foldername}")

    def download_tiles(self, bounding_box: List[float]) -> List[str]:
        """
        Download all tiles within the bounding box
        
        Args:
            bounding_box (List[float]): [left, down, right, top]
            
        Returns:
            List[str]: List of downloaded file paths
        """
        x_tiles, y_tiles = self.tile_indices(bounding_box)
        downloaded_files = []
        total_tiles = (x_tiles[1] - x_tiles[0]) * (y_tiles[1] - y_tiles[0])
        
        logger.info(f"Starting download of {total_tiles} tiles for date {self.date}")
        
        if self.run:
            self.run.log("total_tiles_to_download", total_tiles)
        
        downloaded_count = 0
        for x_index in range(x_tiles[0], x_tiles[1]):
            for y_index in range(y_tiles[0], y_tiles[1]):
                self.mkdir(f"{DOWNLOAD_FOLDER}/{self.layer}")
                filename = f"{DOWNLOAD_FOLDER}/{self.layer}/{self.date}-{x_index}-{y_index}.tif"
                
                if os.path.exists(filename):
                    downloaded_files.append(filename)
                    logger.info(f"File already exists: {filename}")
                else:
                    downloaded_file = self.download_tile(x_index, y_index, filename)
                    if downloaded_file:
                        downloaded_files.append(downloaded_file)
                        downloaded_count += 1
                        
                        # Log progress to Azure ML
                        if self.run and downloaded_count % 10 == 0:
                            progress = (downloaded_count / total_tiles) * 100
                            self.run.log("download_progress_percent", progress)
        
        logger.info(f"Download completed. {len(downloaded_files)} files downloaded.")
        
        if self.run:
            self.run.log("total_files_downloaded", len(downloaded_files))
        
        return downloaded_files

    def register_new_search(self) -> str:
        """
        Register new search with HLS titiler
        
        Returns:
            str: Search ID
        """
        try:
            response = requests.post(
                REGISTER_ENDPOINT,
                headers={"Content-Type": "application/json", "accept": "application/json"},
                data=json.dumps({
                    "datetime": f"{self.date}T00:00:00Z/{self.date}T23:59:59Z",
                    "collections": [self.layer],
                }),
                timeout=30
            )
            response.raise_for_status()
            search_id = response.json()["searchid"]
            
            logger.info(f"Registered new search with ID: {search_id}")
            
            if self.run:
                self.run.log("search_id", search_id)
                
            return search_id
            
        except Exception as e:
            logger.error(f"Failed to register search: {e}")
            raise

    def tile_indices(self, bounding_box: List[float]) -> List[List[int]]:
        """
        Extract tile indices based on bounding_box

        Args:
            bounding_box (List[float]): [left, down, right, top]

        Returns:
            List[List[int]]: [[start_x, end_x], [start_y, end_y]]
        """
        start_x, start_y, _ = TMS.tile(bounding_box[0], bounding_box[3], ZOOM_LEVEL)
        end_x, end_y, _ = TMS.tile(bounding_box[2], bounding_box[1], ZOOM_LEVEL)
        
        tile_ranges = [[start_x, end_x], [start_y, end_y]]
        
        if self.run:
            self.run.log("tile_range_x", f"{start_x}-{end_x}")
            self.run.log("tile_range_y", f"{start_y}-{end_y}")
        
        return tile_ranges

    def get_azure_storage_urls(self, downloaded_files: List[str]) -> List[str]:
        """
        Get Azure Storage URLs for downloaded files
        
        Args:
            downloaded_files (List[str]): List of local file paths
            
        Returns:
            List[str]: List of Azure Storage URLs
        """
        if not self.use_azure_storage or not self.blob_service_client:
            return []
        
        urls = []
        for file_path in downloaded_files:
            filename = os.path.basename(file_path)
            blob_name = f"{self.layer}/{self.date}/{filename}"
            blob_url = f"https://{self.azure_storage_account}.blob.core.windows.net/{self.azure_container}/{blob_name}"
            urls.append(blob_url)
        
        return urls
