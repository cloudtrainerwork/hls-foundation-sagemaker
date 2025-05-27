print("!!!! importing packages")
import os
import matplotlib.pyplot as plt
import mmcv
import gc
import logging
import json
from typing import List, Dict, Any, Optional

import rasterio
import torch

from huggingface_hub import hf_hub_download

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines.compose import Compose
from mmseg.models import build_segmentor

from geospatial_fm.post_process import PostProcess

print("!!!! Done importing packages")

# Configure logging for Azure ML
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure ML compatible paths
CONFIG_DIR = os.path.join(os.getenv("AZUREML_MODEL_DIR", "./"), "configs", "{experiment}_config", "geospatial_fm_config.py")
DOWNLOAD_FOLDER = os.getenv("AZUREML_DATA_DIR", "/tmp/downloads")

# Ensure download folder exists
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

MODEL_CONFIGS = {
    "flood": {
        "config": "sen1floods11_Prithvi_100M.py",
        "repo": "ibm-nasa-geospatial/Prithvi-100M-sen1floods11",
        "weight": "sen1floods11_Prithvi_100M.pth",
        "collections": ["HLSS30"],
    },
    "burn_scars": {
        "config": "burn_scars_Prithvi_100M.py",
        "repo": "ibm-nasa-geospatial/Prithvi-100M-burn-scar",
        "weight": "burn_scars_Prithvi_100M.pth",
        "collections": ["HLSS30", "HLSL30"],
    },
    "prithvi_global": {
        "config": "prithvi_global_300M.py",
        "repo": None,  # Local model for Azure ML
        "weight": "prithvi-global-300M.pt",
        "collections": ["HLSS30", "HLSL30"],
    },
}


def update_config(config_path: str, model_path: str) -> None:
    """Update model configuration with correct path"""
    try:
        with open(config_path, "r") as config_file:
            config_details = config_file.read()
            updated_config = config_details.replace(
                "<path to pretrained weights>", model_path
            )

        with open(config_path, "w") as config_file:
            config_file.write(updated_config)
        
        logger.info(f"Updated config file: {config_path}")
    except Exception as e:
        logger.error(f"Error updating config {config_path}: {str(e)}")
        raise


def load_model(model_name: str) -> 'Infer':
    """Load model from HuggingFace or local Azure ML model directory"""
    try:
        logger.info(f"Loading model: {model_name}")
        
        model_config = MODEL_CONFIGS[model_name]
        
        if model_config['repo']:
            # Download from HuggingFace
            repo = model_config['repo']
            config_path = hf_hub_download(repo, filename=model_config['config'])
            model_path = hf_hub_download(repo, filename=model_config['weight'])
        else:
            # Use local model from Azure ML model directory
            model_dir = os.getenv("AZUREML_MODEL_DIR", "./")
            config_path = os.path.join(model_dir, model_config['config'])
            model_path = os.path.join(model_dir, model_config['weight'])
            
            # Check if local files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Update config with model path
        if os.path.exists(config_path):
            update_config(config_path, model_path)
        
        # Initialize inference engine
        infer = Infer(config_path, model_path)
        _ = infer.load_model()
        
        logger.info(f"Successfully loaded model: {model_name}")
        return infer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise


class AzureMLLoader:
    """Azure ML compatible model loader"""
    
    def __init__(self):
        self.initialized = False
        self.models = {}

    def initialize(self, model_names: Optional[List[str]] = None) -> None:
        """
        Initialize models for Azure ML deployment
        :param model_names: List of model names to load, defaults to all available
        """
        try:
            if model_names is None:
                model_names = list(MODEL_CONFIGS.keys())
            
            logger.info(f"Initializing models: {model_names}")
            
            for model_name in model_names:
                try:
                    self.models[model_name] = load_model(model_name)
                    logger.info(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {str(e)}")
                    # Continue with other models
            
            self.initialized = True
            logger.info(f"Initialization complete. Loaded {len(self.models)} models.")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def handle(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Handle inference request
        :param data: Input data containing image paths and parameters
        :param model_name: Name of the model to use
        :return: Inference results
        """
        try:
            if not self.initialized:
                raise RuntimeError("Loader not initialized")
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            logger.info(f"Running inference with model: {model_name}")
            
            # Extract image paths from data
            image_paths = data.get('images', [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            # Run inference
            model = self.models[model_name]
            results = model.infer(image_paths)
            
            # Post-process results
            processed_results = model.postprocess(results, image_paths)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in handle: {str(e)}")
            raise


class Infer:
    """Enhanced inference class for Azure ML deployment"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.initialized = False
        self.config_filename = config_path
        self.checkpoint_filename = checkpoint_path
        self.model = None
        self.config = None
        self.device = None

    def initialize_azure_ml(self) -> None:
        """Initialize for Azure ML deployment"""
        try:
            self.initialized = True
            logger.info(f"Initializing model with config: {self.config_filename}")
            logger.info(f"Checkpoint: {self.checkpoint_filename}")
            
            # Load model
            self.load_model()
            
        except Exception as e:
            logger.error(f"Error in Azure ML initialization: {str(e)}")
            raise

    def load_model(self) -> torch.nn.Module:
        """Load the segmentation model"""
        try:
            logger.info("Loading model configuration...")
            self.config = mmcv.Config.fromfile(self.config_filename)
            self.config.model.pretrained = None
            self.config.model.train_cfg = None

            # Determine device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            if self.checkpoint_filename is not None:
                logger.info("Initializing segmentor...")
                self.model = init_segmentor(
                    self.config, self.checkpoint_filename, device=str(self.device)
                )
                
                logger.info("Loading checkpoint...")
                self.checkpoint = load_checkpoint(
                    self.model, self.checkpoint_filename, map_location="cpu"
                )
                
                # Set model metadata
                if "meta" in self.checkpoint:
                    self.model.CLASSES = self.checkpoint["meta"].get("CLASSES", ["background", "foreground"])
                    self.model.PALETTE = self.checkpoint["meta"].get("PALETTE", [[0, 0, 0], [255, 255, 255]])
                else:
                    logger.warning("No metadata found in checkpoint, using defaults")
                    self.model.CLASSES = ["background", "foreground"]
                    self.model.PALETTE = [[0, 0, 0], [255, 255, 255]]
                    
            else:
                raise ValueError("Checkpoint filename is required")

            self.model.cfg = self.config  # save the config in the model for convenience
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def infer(self, images: List[str]) -> List[torch.Tensor]:
        """
        Run inference on provided images
        Args:
            images (List[str]): List of image file paths
        Returns:
            List of inference results
        """
        try:
            if not self.model:
                raise RuntimeError("Model not loaded")
            
            logger.info(f"Running inference on {len(images)} images")
            
            # Prepare test pipeline
            test_pipeline = self.config.data.test.pipeline
            test_pipeline = Compose(test_pipeline)
            
            data = []
            for image_path in images:
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                    
                image_data = dict(img_info=dict(filename=os.path.basename(image_path)))
                image_data["seg_fields"] = []
                image_data["img_prefix"] = os.path.dirname(image_path) or DOWNLOAD_FOLDER
                image_data["seg_prefix"] = os.path.dirname(image_path) or DOWNLOAD_FOLDER
                image_data = test_pipeline(image_data)
                data.append(image_data)

            if not data:
                logger.warning("No valid images found for inference")
                return []

            # Collate data
            data = collate(data, samples_per_gpu=len(data))
            
            # Move to device if using GPU
            if next(self.model.parameters()).is_cuda:
                data = scatter(data, [self.device])[0]
            else:
                data["img_metas"] = [i.data[0] for i in list(data["img_metas"])]

            # Run inference
            with torch.no_grad():
                results = self.model(return_loss=False, rescale=True, **data)
            
            logger.info(f"Inference completed on {len(results)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise

    def postprocess(self, results: List[torch.Tensor], files: List[str]) -> Dict[str, Any]:
        """
        Postprocess results to prepare GeoJSON based on the images
        
        :param results: List of results from infer method
        :param files: List of files on which the inference was performed
        :return: GeoJSON of detected features
        """
        try:
            logger.info(f"Post-processing {len(results)} results")
            
            transforms = []
            geojson_list = []
            
            # Extract transforms from raster files
            for tile_path in files:
                try:
                    with rasterio.open(tile_path) as raster:
                        transforms.append(raster.transform)
                except Exception as e:
                    logger.warning(f"Could not read transform from {tile_path}: {str(e)}")
                    continue
            
            # Process each result
            for index, result in enumerate(results):
                if index >= len(transforms):
                    logger.warning(f"No transform available for result {index}")
                    continue
                    
                try:
                    # Extract shapes from predictions
                    detections = PostProcess.extract_shapes(result, transforms[index])
                    
                    # Remove intersecting detections
                    detections = PostProcess.remove_intersections(detections)
                    
                    # Convert to GeoJSON
                    geojson = PostProcess.convert_to_geojson(detections)
                    
                    # Convert coordinate systems
                    for geometry in geojson:
                        updated_geometry = PostProcess.convert_geojson(geometry)
                        geojson_list.append(updated_geometry)
                        
                except Exception as e:
                    logger.warning(f"Error processing result {index}: {str(e)}")
                    continue
            
            result_geojson = {
                "type": "FeatureCollection", 
                "features": geojson_list,
                "metadata": {
                    "total_results": len(results),
                    "processed_features": len(geojson_list)
                }
            }
            
            logger.info(f"Post-processing complete. Generated {len(geojson_list)} features.")
            return result_geojson
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            raise


# Global service instance for Azure ML
logger.info("Initializing global Azure ML service")
_service = AzureMLLoader()


def init():
    """Initialize function for Azure ML scoring script"""
    try:
        logger.info("Azure ML init() called")
        if not _service.initialized:
            _service.initialize()
        logger.info("Azure ML initialization complete")
    except Exception as e:
        logger.error(f"Error in Azure ML init: {str(e)}")
        raise


def run(data: str) -> str:
    """Run function for Azure ML scoring script"""
    try:
        logger.info("Azure ML run() called")
        
        # Parse input data
        input_data = json.loads(data) if isinstance(data, str) else data
        
        # Extract parameters
        model_name = input_data.get('model_id', 'burn_scars')
        images = input_data.get('images', [])
        
        if not images:
            raise ValueError("No images provided for inference")
        
        # Run inference
        results = _service.handle({'images': images}, model_name)
        
        return json.dumps(results)
        
    except Exception as e:
        logger.error(f"Error in Azure ML run: {str(e)}")
        return json.dumps({"error": str(e)})


def handle(data, context=None):
    """Legacy handle function for compatibility"""
    try:
        logger.info(f"Legacy handle called with data: {data}")
        
        if not _service.initialized:
            _service.initialize()
            
        if data is None:
            return None
            
        # For legacy compatibility, assume burn_scars model
        return _service.handle(data, 'burn_scars')
        
    except Exception as e:
        logger.error(f"Error in legacy handle: {str(e)}")
        raise


print("!!!!! Model loading preparation complete")
