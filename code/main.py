import gc
import os
import rasterio
import time
import torch
import json
import logging
from typing import Dict, Any, List

# Azure ML specific imports
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Application imports - updated paths for Azure ML
from geospatial_fm.downloader import Downloader
from geospatial_fm.infer import Infer
from geospatial_fm.post_process import PostProcess

from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from huggingface_hub import hf_hub_download
from multiprocessing import Pool, cpu_count

# Configure logging for Azure ML
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Geospatial Foundation Model API", version="1.0.0")

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Updated model configurations for Azure ML
MODEL_CONFIGS = {
    'flood': {
        'config': 'sen1floods11_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-sen1floods11',
        'weight': 'sen1floods11_Prithvi_100M.pth',
        'collections': ['HLSS30'],
    },
    'burn_scars': {
        'config': 'burn_scars_Prithvi_100M.py',
        'repo': 'ibm-nasa-geospatial/Prithvi-100M-burn-scar',
        'weight': 'burn_scars_Prithvi_100M.pth',
        'collections': ['HLSS30', 'HLSL30'],
    },
    'prithvi_global': {
        'config': 'prithvi_global_300M.py',
        'repo': None,  # Local model
        'weight': 'prithvi-global-300M.pt',
        'collections': ['HLSS30', 'HLSL30'],
    }
}


def update_config(config_path: str, model_path: str) -> None:
    """Update model configuration with correct path"""
    try:
        with open(config_path, 'r') as config_file:
            config_details = config_file.read()
            updated_config = config_details.replace('<path to pretrained weights>', model_path)

        with open(config_path, 'w') as config_file:
            config_file.write(updated_config)
        
        logger.info(f"Updated config file: {config_path}")
    except Exception as e:
        logger.error(f"Error updating config {config_path}: {str(e)}")
        raise


def load_model(model_name: str) -> Infer:
    """Load model from HuggingFace or local path"""
    try:
        logger.info(f"Loading model: {model_name}")
        
        model_config = MODEL_CONFIGS[model_name]
        
        if model_config['repo']:
            # Download from HuggingFace
            repo = model_config['repo']
            config_path = hf_hub_download(repo, filename=model_config['config'])
            model_path = hf_hub_download(repo, filename=model_config['weight'])
        else:
            # Use local model (for prithvi_global)
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


def download_files(infer_date: str, layer: str, bounding_box: List[float]) -> List[str]:
    """Download satellite imagery files"""
    try:
        downloader = Downloader(infer_date, layer)
        return downloader.download_tiles(bounding_box)
    except Exception as e:
        logger.error(f"Error downloading files for {layer}: {str(e)}")
        return []


# Initialize models - with error handling for Azure ML startup
MODELS = {}
for model_name in MODEL_CONFIGS:
    try:
        MODELS[model_name] = load_model(model_name)
        logger.info(f"Model {model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        # Continue with other models rather than failing entirely


# Azure ML health check endpoint
@app.get("/health", status_code=200)
def health():
    """Health check endpoint for Azure ML"""
    return {
        'status': 'healthy',
        'models_loaded': list(MODELS.keys()),
        'timestamp': time.time()
    }


# Legacy health check for compatibility
@app.get("/", status_code=200)
def root():
    """Root endpoint"""
    return health()


@app.get('/models')
def list_models():
    """List available models"""
    try:
        available_models = list(MODELS.keys())
        response = jsonable_encoder(available_models)
        return JSONResponse({
            'models': response,
            'total_count': len(available_models)
        })
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing models")


@app.post("/predict")
async def infer_from_model(request: Request):
    """Main prediction endpoint for Azure ML"""
    try:
        body = await request.json()
        logger.info(f"Received prediction request: {json.dumps(body, indent=2)}")

        # Parse request body - handle both Azure ML and custom formats
        if 'instances' in body:
            instances = body['instances']
            if isinstance(instances, list):
                instances = instances[0]
        else:
            instances = body

        # Extract parameters
        model_id = instances.get('model_id')
        infer_date = instances.get('date')
        bounding_box = instances.get('bounding_box')

        # Validate inputs
        if not model_id:
            raise HTTPException(status_code=422, detail="model_id is required")
        if not infer_date:
            raise HTTPException(status_code=422, detail="date is required")
        if not bounding_box:
            raise HTTPException(status_code=422, detail="bounding_box is required")

        if model_id not in MODELS:
            raise HTTPException(
                status_code=422, 
                detail=f"Model {model_id} not available. Available models: {list(MODELS.keys())}"
            )

        # Get model inference engine
        infer = MODELS[model_id]
        all_tiles = []
        geojson_list = []
        download_infos = []

        # Prepare download tasks
        for layer in MODEL_CONFIGS[model_id]['collections']:
            download_infos.append((infer_date, layer, bounding_box))

        # Download satellite imagery tiles
        logger.info("Starting tile downloads...")
        start_time = time.time()
        
        try:
            # Use multiprocessing for parallel downloads
            with Pool(min(cpu_count() - 1, len(download_infos))) as pool:
                all_tiles_nested = pool.starmap(download_files, download_infos)
                all_tiles = [tile for tiles in all_tiles_nested for tile in tiles]
        except Exception as download_error:
            logger.error(f"Error during parallel downloads: {str(download_error)}")
            # Fallback to sequential downloads
            for download_info in download_infos:
                tiles = download_files(*download_info)
                all_tiles.extend(tiles)

        download_time = time.time() - start_time
        logger.info(f"Download completed in {download_time:.2f} seconds. Downloaded {len(all_tiles)} tiles.")

        # Run inference if tiles were downloaded
        if all_tiles:
            logger.info("Starting model inference...")
            start_time = time.time()
            
            try:
                # Run model inference
                results = infer.infer(all_tiles)
                
                # Extract transforms for coordinate conversion
                transforms = []
                for tile in all_tiles:
                    try:
                        with rasterio.open(tile) as raster:
                            transforms.append(raster.transform)
                    except Exception as transform_error:
                        logger.warning(f"Could not read transform from {tile}: {str(transform_error)}")
                        continue

                # Post-process results
                for index, result in enumerate(results):
                    if index < len(transforms):
                        try:
                            # Extract and process shapes
                            detections = PostProcess.extract_shapes(result, transforms[index])
                            detections = PostProcess.remove_intersections(detections)
                            geojson = PostProcess.convert_to_geojson(detections)
                            
                            # Convert coordinate systems
                            for geometry in geojson:
                                updated_geometry = PostProcess.convert_geojson(geometry)
                                geojson_list.append(updated_geometry)
                                
                        except Exception as process_error:
                            logger.warning(f"Error processing result {index}: {str(process_error)}")
                            continue

                infer_time = time.time() - start_time
                logger.info(f"Inference completed in {infer_time:.2f} seconds. Found {len(geojson_list)} features.")
                
            except Exception as inference_error:
                logger.error(f"Error during inference: {str(inference_error)}")
                raise HTTPException(status_code=500, detail="Error during model inference")
        else:
            logger.warning("No tiles downloaded, returning empty results")

        # Cleanup
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {str(cleanup_error)}")

        # Format response
        final_geojson = {
            'predictions': [{
                'type': 'FeatureCollection', 
                'features': geojson_list
            }],
            'metadata': {
                'model_id': model_id,
                'date': infer_date,
                'bounding_box': bounding_box,
                'tiles_processed': len(all_tiles),
                'features_found': len(geojson_list),
                'processing_time': {
                    'download_seconds': download_time,
                    'inference_seconds': infer_time if 'infer_time' in locals() else 0
                }
            }
        }

        return JSONResponse(content=jsonable_encoder(final_geojson))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# For Azure ML scoring script compatibility
def init():
    """Initialize function called by Azure ML"""
    logger.info("Initializing Azure ML scoring service...")
    # Models are already loaded at startup
    logger.info("Initialization complete")


def run(raw_data: str) -> str:
    """Run function called by Azure ML scoring script"""
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Create a mock request object
        class MockRequest:
            async def json(self):
                return data
        
        # Use the main prediction function
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(infer_from_model(MockRequest()))
        return response.body.decode()
        
    except Exception as e:
        logger.error(f"Error in run function: {str(e)}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
