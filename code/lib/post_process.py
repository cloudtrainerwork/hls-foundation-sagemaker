import cv2
import numpy as np
import rasterio
import rasterio.warp
import logging
import json
import os
from typing import List, Tuple, Dict, Any, Optional, Union
import time

from geojson import Feature, Polygon, FeatureCollection
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from scipy.interpolate import splprep, splev
from shapely import geometry

# Azure ML specific imports
from azureml.core import Run
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Configuration constants
AREA_THRESHOLD = 0.05
MIN_POINTS = 3
PREDICT_THRESHOLD = 0.5
BLUR_FACTOR = (15, 15)
BLUR_THRESHOLD = 127

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostProcess:
    def __init__(self, 
                 area_threshold: float = AREA_THRESHOLD,
                 min_points: int = MIN_POINTS,
                 predict_threshold: float = PREDICT_THRESHOLD,
                 blur_factor: Tuple[int, int] = BLUR_FACTOR,
                 blur_threshold: int = BLUR_THRESHOLD,
                 azure_storage_account: Optional[str] = None,
                 azure_container: Optional[str] = None):
        """
        Initialize PostProcess with configurable parameters and Azure ML integration
        
        Args:
            area_threshold: Minimum area threshold for polygon filtering
            min_points: Minimum number of points for contour processing
            predict_threshold: Threshold for binary prediction conversion
            blur_factor: Blur kernel size for smoothing
            blur_threshold: Threshold after blurring
            azure_storage_account: Azure Storage Account name (optional)
            azure_container: Azure Blob Container name (optional)
        """
        self.area_threshold = area_threshold
        self.min_points = min_points
        self.predict_threshold = predict_threshold
        self.blur_factor = blur_factor
        self.blur_threshold = blur_threshold
        
        # Azure ML integration
        try:
            self.run = Run.get_context()
            logger.info("Azure ML run context initialized")
        except Exception:
            self.run = None
            logger.info("Not running in Azure ML context")
        
        # Azure Storage integration
        self.azure_storage_account = azure_storage_account
        self.azure_container = azure_container
        self.blob_service_client = None
        
        if azure_storage_account:
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
        
        # Log configuration to Azure ML
        if self.run:
            self.run.log("area_threshold", self.area_threshold)
            self.run.log("min_points", self.min_points)
            self.run.log("predict_threshold", self.predict_threshold)
            self.run.log("blur_factor", str(self.blur_factor))
            self.run.log("blur_threshold", self.blur_threshold)

    def prepare_bitmap(self, predictions: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Prepare bitmap from predictions with Azure ML logging
        
        Args:
            predictions: Raw prediction array
            width: Target width
            height: Target height
            
        Returns:
            np.ndarray: Reshaped predictions
        """
        start_time = time.time()
        
        try:
            predictions = predictions.reshape((height, width))
            logger.info(f"Bitmap prepared: {width}x{height}")
            
            if self.run:
                self.run.log("bitmap_width", width)
                self.run.log("bitmap_height", height)
                self.run.log("bitmap_shape", f"{width}x{height}")
                processing_time = time.time() - start_time
                self.run.log("bitmap_preparation_time", processing_time)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to prepare bitmap: {e}")
            if self.run:
                self.run.log("bitmap_preparation_error", str(e))
            raise

    def extract_shapes(self, predictions: np.ndarray, transform: Any) -> List[Tuple[np.ndarray, float]]:
        """
        Extract shapes from predictions with enhanced Azure ML logging
        
        Args:
            predictions: Prediction array
            transform: Rasterio transform object
            
        Returns:
            List[Tuple[np.ndarray, float]]: List of (coordinates, score) tuples
        """
        start_time = time.time()
        
        try:
            # Create binary bitmap
            bitmap = (predictions > self.predict_threshold).astype(dtype="uint8") * 255
            logger.info(f"Created binary bitmap with threshold {self.predict_threshold}")
            
            # Apply blur and threshold
            img_blurred = cv2.blur(bitmap, self.blur_factor)
            thresholded_img = (img_blurred > self.blur_threshold).astype(dtype="uint8") * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                np.asarray(thresholded_img, dtype="uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            
            logger.info(f"Found {len(contours)} contours")
            
            shape = bitmap.shape
            smoothened = []
            valid_contours = 0
            
            for i, contour in enumerate(contours):
                length = len(contour)
                if length > self.min_points:
                    try:
                        # Extract coordinates
                        y, x = contour.T
                        x = x.tolist()[0]
                        y = y.tolist()[0]
                        
                        # Smooth using spline interpolation
                        knots_vector, params = splprep([x, y], s=3.0, quiet=1)
                        new_params = np.linspace(params.min(), params.max(), 25)
                        x_new, y_new = splev(new_params, knots_vector, der=0, ext=3)
                        
                        # Convert to polygon and calculate score
                        new_polygon = []
                        res_array = []
                        
                        for pair in zip(x_new, y_new):
                            pair = list(pair)
                            # Boundary clipping
                            pair[0] = max(0, min(pair[0], shape[0] - 1))
                            pair[1] = max(0, min(pair[1], shape[1] - 1))
                            
                            new_polygon.append((pair[0], pair[1]))
                            res_array.append(self.convert_xy_to_latlon(pair[0], pair[1], transform))
                        
                        # Calculate polygon score
                        img = Image.new("L", (shape[0], shape[1]), 0)
                        ImageDraw.Draw(img).polygon(new_polygon, outline=1, fill=1)
                        mask = np.where(np.array(img).T > 0)
                        score = predictions[mask]
                        
                        if len(score) > 0:
                            score_value = float(np.mean(score))
                            res_array.append(res_array[0])  # Close polygon
                            smoothened.append([np.asarray(res_array), score_value])
                            valid_contours += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process contour {i}: {e}")
                        continue
            
            processing_time = time.time() - start_time
            logger.info(f"Shape extraction completed: {valid_contours} valid shapes from {len(contours)} contours")
            
            # Log metrics to Azure ML
            if self.run:
                self.run.log("total_contours_found", len(contours))
                self.run.log("valid_shapes_extracted", valid_contours)
                self.run.log("shape_extraction_time", processing_time)
                self.run.log("contour_processing_success_rate", valid_contours / max(len(contours), 1))
            
            return smoothened
            
        except Exception as e:
            logger.error(f"Failed to extract shapes: {e}")
            if self.run:
                self.run.log("shape_extraction_error", str(e))
            raise

    def convert_to_geojson(self, shapes: List[Tuple[np.ndarray, float]], 
                          additional_properties: Optional[Dict] = None) -> List[Feature]:
        """
        Convert shapes to GeoJSON features with enhanced properties
        
        Args:
            shapes: List of (coordinates, score) tuples
            additional_properties: Additional properties to add to each feature
            
        Returns:
            List[Feature]: List of GeoJSON features
        """
        start_time = time.time()
        
        try:
            geojson_features = []
            additional_properties = additional_properties or {}
            
            for id_, shape in enumerate(shapes):
                try:
                    coordinates = shape[0]
                    score = float(shape[1])
                    
                    # Create properties
                    properties = {
                        "id": id_,
                        "score": score,
                        "area": self._calculate_polygon_area(coordinates),
                        "perimeter": self._calculate_polygon_perimeter(coordinates),
                        **additional_properties
                    }
                    
                    # Create GeoJSON feature
                    feature = Feature(
                        properties=properties,
                        geometry=Polygon([[[float(lon), float(lat)] for lon, lat in coordinates]])
                    )
                    
                    geojson_features.append(feature)
                    
                except Exception as e:
                    logger.warning(f"Failed to create GeoJSON feature {id_}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            logger.info(f"Created {len(geojson_features)} GeoJSON features")
            
            if self.run:
                self.run.log("geojson_features_created", len(geojson_features))
                self.run.log("geojson_conversion_time", processing_time)
                if geojson_features:
                    scores = [f["properties"]["score"] for f in geojson_features]
                    self.run.log("average_feature_score", np.mean(scores))
                    self.run.log("max_feature_score", np.max(scores))
                    self.run.log("min_feature_score", np.min(scores))
            
            return geojson_features
            
        except Exception as e:
            logger.error(f"Failed to convert to GeoJSON: {e}")
            if self.run:
                self.run.log("geojson_conversion_error", str(e))
            raise

    def convert_geojson(self, results: Dict) -> Dict:
        """
        Convert GeoJSON from Web Mercator to WGS84
        
        Args:
            results: GeoJSON-like dictionary
            
        Returns:
            Dict: Converted GeoJSON
        """
        try:
            feature = results["geometry"]
            feature_proj = rasterio.warp.transform_geom(
                CRS.from_epsg(3857), CRS.from_epsg(4326), feature
            )
            results["geometry"] = feature_proj
            
            if self.run:
                self.run.log("geojson_projection_converted", True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to convert GeoJSON projection: {e}")
            if self.run:
                self.run.log("geojson_projection_error", str(e))
            raise

    def remove_intersections(self, shapes: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Remove intersecting polygons, keeping larger ones with Azure ML logging
        
        Args:
            shapes: List of (coordinates, score) tuples
            
        Returns:
            np.ndarray: Filtered shapes
        """
        start_time = time.time()
        
        try:
            computed_polygons = []
            selected_indices = []
            selected_shapes = []
            areas = []
            
            # Filter valid polygons by area
            for i, shape in enumerate(shapes):
                try:
                    polygon = geometry.Polygon(shape[0])
                    if polygon.is_valid:
                        area = polygon.area
                        if area > self.area_threshold:
                            computed_polygons.append(polygon)
                            areas.append(area)
                            selected_shapes.append(shape)
                except Exception as e:
                    logger.warning(f"Invalid polygon at index {i}: {e}")
                    continue
            
            logger.info(f"Filtered to {len(selected_shapes)} valid polygons from {len(shapes)} total")
            
            # Remove intersections by area priority
            if len(areas) > 0:
                computed_polygons = np.asarray(computed_polygons)
                polygon_indices = list(np.argsort(areas))
                
                while len(polygon_indices) > 0:
                    selected_index = polygon_indices[-1]  # Largest remaining polygon
                    selected_polygon = computed_polygons[selected_index]
                    selected_indices.append(selected_index)
                    polygon_indices.remove(selected_index)
                    
                    # Remove intersecting polygons
                    indices_holder = polygon_indices.copy()
                    removed_count = 0
                    for index in indices_holder:
                        if computed_polygons[index].intersects(selected_polygon):
                            polygon_indices.remove(index)
                            removed_count += 1
                    
                    if removed_count > 0:
                        logger.debug(f"Removed {removed_count} intersecting polygons")
            
            final_shapes = np.array(selected_shapes, dtype='object')[selected_indices] if selected_indices else np.array([])
            
            processing_time = time.time() - start_time
            logger.info(f"Intersection removal completed: {len(final_shapes)} final shapes")
            
            # Log metrics to Azure ML
            if self.run:
                self.run.log("shapes_before_filtering", len(shapes))
                self.run.log("shapes_after_area_filter", len(selected_shapes))
                self.run.log("shapes_after_intersection_removal", len(final_shapes))
                self.run.log("intersection_removal_time", processing_time)
                if len(shapes) > 0:
                    retention_rate = len(final_shapes) / len(shapes)
                    self.run.log("shape_retention_rate", retention_rate)
            
            return final_shapes
            
        except Exception as e:
            logger.error(f"Failed to remove intersections: {e}")
            if self.run:
                self.run.log("intersection_removal_error", str(e))
            raise

    def convert_xy_to_latlon(self, row: float, col: float, transform: Any) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon with error handling
        
        Args:
            row: Row coordinate
            col: Column coordinate
            transform: Rasterio transform object
            
        Returns:
            Tuple[float, float]: (longitude, latitude)
        """
        try:
            transform = rasterio.transform.guard_transform(transform)
            return rasterio.transform.xy(transform, row, col, offset="center")
        except Exception as e:
            logger.error(f"Failed to convert coordinates ({row}, {col}): {e}")
            # Return a default coordinate or raise
            raise

    def process_predictions(self, 
                          predictions: np.ndarray, 
                          transform: Any,
                          width: int, 
                          height: int,
                          output_file: Optional[str] = None,
                          upload_to_azure: bool = False) -> FeatureCollection:
        """
        Complete processing pipeline from predictions to GeoJSON
        
        Args:
            predictions: Raw predictions
            transform: Rasterio transform
            width: Image width
            height: Image height
            output_file: Path to save GeoJSON (optional)
            upload_to_azure: Whether to upload to Azure Storage
            
        Returns:
            FeatureCollection: Complete GeoJSON FeatureCollection
        """
        start_time = time.time()
        logger.info("Starting complete post-processing pipeline")
        
        try:
            # Step 1: Prepare bitmap
            bitmap = self.prepare_bitmap(predictions, width, height)
            
            # Step 2: Extract shapes
            shapes = self.extract_shapes(bitmap, transform)
            
            # Step 3: Remove intersections
            filtered_shapes = self.remove_intersections(shapes)
            
            # Step 4: Convert to GeoJSON
            features = self.convert_to_geojson(filtered_shapes)
            
            # Create FeatureCollection
            feature_collection = FeatureCollection(features)
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(feature_collection, f, indent=2)
                logger.info(f"GeoJSON saved to: {output_file}")
                
                # Upload to Azure Storage if requested
                if upload_to_azure and self.blob_service_client and self.azure_container:
                    self._upload_to_azure_storage(output_file)
            
            total_time = time.time() - start_time
            logger.info(f"Post-processing pipeline completed in {total_time:.2f} seconds")
            
            # Final logging to Azure ML
            if self.run:
                self.run.log("total_processing_time", total_time)
                self.run.log("final_feature_count", len(features))
                self.run.log("processing_pipeline_completed", True)
            
            return feature_collection
            
        except Exception as e:
            logger.error(f"Post-processing pipeline failed: {e}")
            if self.run:
                self.run.log("processing_pipeline_error", str(e))
            raise

    def _upload_to_azure_storage(self, file_path: str):
        """Upload file to Azure Blob Storage"""
        try:
            blob_name = f"geojson/{os.path.basename(file_path)}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.azure_container, 
                blob=blob_name
            )
            
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Uploaded {file_path} to Azure Storage as {blob_name}")
            
            if self.run:
                self.run.log("azure_upload_completed", True)
                self.run.log("azure_blob_name", blob_name)
                
        except Exception as e:
            logger.error(f"Failed to upload to Azure Storage: {e}")
            if self.run:
                self.run.log("azure_upload_error", str(e))

    def _calculate_polygon_area(self, coordinates: np.ndarray) -> float:
        """Calculate polygon area using Shapely"""
        try:
            polygon = geometry.Polygon(coordinates)
            return float(polygon.area)
        except Exception:
            return 0.0

    def _calculate_polygon_perimeter(self, coordinates: np.ndarray) -> float:
        """Calculate polygon perimeter using Shapely"""
        try:
            polygon = geometry.Polygon(coordinates)
            return float(polygon.length)
        except Exception:
            return 0.0

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for monitoring"""
        return {
            "area_threshold": self.area_threshold,
            "min_points": self.min_points,
            "predict_threshold": self.predict_threshold,
            "blur_factor": self.blur_factor,
            "blur_threshold": self.blur_threshold,
            "azure_storage_enabled": self.blob_service_client is not None,
            "azure_ml_context": self.run is not None
        }
