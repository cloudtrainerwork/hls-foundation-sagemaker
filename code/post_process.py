import cv2
import numpy as np
import rasterio
import rasterio.warp
import logging

from geojson import Feature, Polygon
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from scipy.interpolate import splprep, splev
from shapely import geometry

# Configure logging for Azure ML
logger = logging.getLogger(__name__)

# Configuration constants
AREA_THRESHOLD = 0.05
MIN_POINTS = 3
PREDICT_THRESHOLD = 0.5
BLUR_FACTOR = (15, 15)
BLUR_THRESHOLD = 127


class PostProcess:
    @classmethod
    def prepare_bitmap(cls, predictions, width, height):
        """Prepare bitmap from model predictions"""
        try:
            predictions = predictions.reshape((height, width))
            logger.info(f"Reshaped predictions to {predictions.shape}")
            return predictions
        except Exception as e:
            logger.error(f"Error in prepare_bitmap: {str(e)}")
            raise

    @classmethod
    def extract_shapes(cls, predictions, transform):
        """Extract shapes from predictions using contour detection and smoothing"""
        try:
            # Convert predictions to binary bitmap
            bitmap = (predictions > PREDICT_THRESHOLD).astype(dtype="uint8") * 255
            
            # Apply blur and threshold
            img_blurred = cv2.blur(bitmap, BLUR_FACTOR)
            thresholded_img = (img_blurred > BLUR_THRESHOLD).astype(dtype="uint8") * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                np.asarray(thresholded_img, dtype="uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            shape = bitmap.shape
            smoothened = list()
            
            for contour in contours:
                length = len(contour)
                if length > MIN_POINTS:
                    # Extract x, y coordinates
                    y, x = contour.T
                    x = x.tolist()[0]
                    y = y.tolist()[0]
                    
                    # Smooth the contour using spline interpolation
                    try:
                        knots_vector, params = splprep([x, y], s=3.0, quiet=1)
                        new_params = np.linspace(params.min(), params.max(), 25)
                        x_new, y_new = splev(new_params, knots_vector, der=0, ext=3)
                    except Exception as spline_error:
                        logger.warning(f"Spline interpolation failed: {spline_error}")
                        continue
                    
                    # Process smoothed coordinates
                    new_polygon = list()
                    res_array = list()
                    
                    for pair in zip(x_new, y_new):
                        pair = list(pair)
                        
                        # Boundary checking and correction
                        pair[0] = max(0, min(pair[0], shape[0] - 1))
                        pair[1] = max(0, min(pair[1], shape[1] - 1))
                        
                        new_polygon.append((pair[0], pair[1]))
                        
                        # Convert to lat/lon coordinates
                        try:
                            latlon = cls.convert_xy_to_latlon(pair[0], pair[1], transform)
                            res_array.append(latlon)
                        except Exception as coord_error:
                            logger.warning(f"Coordinate conversion failed: {coord_error}")
                            continue
                    
                    if len(res_array) > 0:
                        # Calculate confidence score
                        img = Image.new("L", (shape[0], shape[1]), 0)
                        ImageDraw.Draw(img).polygon(new_polygon, outline=1, fill=1)
                        mask = np.where(np.array(img).T > 0)
                        
                        if len(mask[0]) > 0:
                            score = predictions[mask]
                            score_length = len(score)
                            if score_length > 0:
                                score = sum(score) / score_length
                            else:
                                score = 0.0
                        else:
                            score = 0.0
                        
                        # Close the polygon
                        res_array.append(res_array[0])
                        smoothened.append([np.asarray(res_array), score])
            
            logger.info(f"Extracted {len(smoothened)} shapes")
            return smoothened
            
        except Exception as e:
            logger.error(f"Error in extract_shapes: {str(e)}")
            raise

    @classmethod
    def convert_to_geojson(cls, shapes):
        """Convert shapes to GeoJSON format"""
        try:
            geojson_dict = []
            for id_, shape in enumerate(shapes):
                try:
                    feature = Feature(
                        properties={
                            "score": float(shape[1]),
                            "id": id_
                        },
                        geometry=Polygon(
                            [[[float(lon), float(lat)] for lon, lat in shape[0]]]
                        ),
                    )
                    geojson_dict.append(feature)
                except Exception as feature_error:
                    logger.warning(f"Failed to convert shape {id_} to GeoJSON: {feature_error}")
                    continue
            
            logger.info(f"Converted {len(geojson_dict)} shapes to GeoJSON")
            return geojson_dict
            
        except Exception as e:
            logger.error(f"Error in convert_to_geojson: {str(e)}")
            raise

    @classmethod
    def convert_geojson(cls, results):
        """Convert GeoJSON coordinates between projections"""
        try:
            feature = results["geometry"]
            feature_proj = rasterio.warp.transform_geom(
                CRS.from_epsg(3857), CRS.from_epsg(4326), feature
            )
            results["geometry"] = feature_proj
            return results
        except Exception as e:
            logger.error(f"Error in convert_geojson: {str(e)}")
            raise

    @classmethod
    def remove_intersections(cls, shapes):
        """Remove intersecting polygons, keeping larger ones"""
        try:
            computed_polygons = list()
            selected_indices = list()
            selected_shapes = list()
            areas = list()
            
            # Filter valid polygons above area threshold
            for shape in shapes:
                try:
                    polygon = geometry.Polygon(shape[0])
                    if polygon.is_valid:
                        area = polygon.area
                        if area > AREA_THRESHOLD:
                            computed_polygons.append(polygon)
                            areas.append(area)
                            selected_shapes.append(shape)
                except Exception as poly_error:
                    logger.warning(f"Invalid polygon skipped: {poly_error}")
                    continue
            
            if len(areas) > 0:
                computed_polygons = np.asarray(computed_polygons)
                polygon_indices = list(np.argsort(areas))
                
                # Select non-intersecting polygons, preferring larger ones
                while len(polygon_indices) > 0:
                    selected_index = polygon_indices[-1]
                    selected_polygon = computed_polygons[selected_index]
                    selected_indices.append(selected_index)
                    polygon_indices.remove(selected_index)
                    
                    # Remove intersecting polygons
                    indices_holder = polygon_indices.copy()
                    for index in indices_holder:
                        try:
                            if computed_polygons[index].intersects(selected_polygon):
                                polygon_indices.remove(index)
                        except Exception as intersect_error:
                            logger.warning(f"Intersection check failed: {intersect_error}")
                            continue
            
            result = np.array(selected_shapes)[selected_indices] if selected_indices else np.array([])
            logger.info(f"Filtered to {len(result)} non-intersecting shapes")
            return result
            
        except Exception as e:
            logger.error(f"Error in remove_intersections: {str(e)}")
            raise

    @classmethod
    def convert_xy_to_latlon(cls, row, col, transform):
        """
        Convert image row, col coordinates to lat, lon using rasterio transform
        """
        try:
            transform = rasterio.transform.guard_transform(transform)
            return rasterio.transform.xy(transform, row, col, offset="center")
        except Exception as e:
            logger.error(f"Error in convert_xy_to_latlon: {str(e)}")
            raise

    @classmethod
    def process_predictions(cls, predictions, transform, width, height):
        """
        Complete processing pipeline for Azure ML deployment
        """
        try:
            logger.info("Starting post-processing pipeline")
            
            # Prepare bitmap
            bitmap = cls.prepare_bitmap(predictions, width, height)
            
            # Extract shapes
            shapes = cls.extract_shapes(bitmap, transform)
            
            # Remove intersections
            filtered_shapes = cls.remove_intersections(shapes)
            
            # Convert to GeoJSON
            geojson_features = cls.convert_to_geojson(filtered_shapes)
            
            logger.info("Post-processing pipeline completed successfully")
            return geojson_features
            
        except Exception as e:
            logger.error(f"Error in process_predictions: {str(e)}")
            raise
