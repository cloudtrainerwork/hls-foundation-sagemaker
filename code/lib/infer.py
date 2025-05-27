import matplotlib.pyplot as plt
import mmcv
import torch
import os
import logging
import json
from typing import List, Optional, Union, Tuple
import numpy as np

from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor

# Azure ML specific imports
from azureml.core import Run, Model, Workspace
from azureml.core.model import InferenceConfig
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from .downloader import DOWNLOAD_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Infer:
    def __init__(self, 
                 config: str, 
                 checkpoint: str,
                 workspace: Optional[Workspace] = None,
                 model_name: Optional[str] = None,
                 model_version: Optional[str] = None,
                 use_azure_model: bool = False):
        """
        Initialize inference class with Azure ML integration
        
        Args:
            config (str): Path to MMSeg config file or Azure ML model config
            checkpoint (str): Path to checkpoint file or Azure ML model checkpoint
            workspace (Workspace): Azure ML workspace (optional)
            model_name (str): Name of registered Azure ML model (optional)
            model_version (str): Version of registered Azure ML model (optional)
            use_azure_model (bool): Whether to load model from Azure ML Model Registry
        """
        self.config_filename = config
        self.checkpoint_filename = checkpoint
        self.workspace = workspace
        self.model_name = model_name
        self.model_version = model_version
        self.use_azure_model = use_azure_model
        
        # Initialize Azure ML run context if available
        try:
            self.run = Run.get_context()
            logger.info("Azure ML run context initialized")
            
            # Get workspace from run context if not provided
            if not self.workspace and hasattr(self.run, 'experiment'):
                self.workspace = self.run.experiment.workspace
                logger.info("Retrieved workspace from run context")
                
        except Exception:
            self.run = None
            logger.info("Not running in Azure ML context")
        
        self.model = None
        self.config = None
        self.device = None

    def download_azure_model_files(self) -> Tuple[str, str]:
        """
        Download model files from Azure ML Model Registry
        
        Returns:
            Tuple[str, str]: (config_path, checkpoint_path)
        """
        if not self.workspace or not self.model_name:
            raise ValueError("Workspace and model_name required for Azure ML model download")
        
        try:
            # Get the registered model
            model = Model(self.workspace, name=self.model_name, version=self.model_version)
            logger.info(f"Retrieved model {self.model_name} version {self.model_version or 'latest'}")
            
            # Download model files to local directory
            model_path = model.download(target_dir="./azure_model", exist_ok=True)
            logger.info(f"Model downloaded to: {model_path}")
            
            # Look for config and checkpoint files
            config_path = None
            checkpoint_path = None
            
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.py') and 'config' in file.lower():
                        config_path = file_path
                    elif file.endswith(('.pth', '.pt', '.ckpt')):
                        checkpoint_path = file_path
            
            if not config_path or not checkpoint_path:
                raise FileNotFoundError("Could not find config or checkpoint files in downloaded model")
            
            logger.info(f"Found config: {config_path}")
            logger.info(f"Found checkpoint: {checkpoint_path}")
            
            if self.run:
                self.run.log("model_download_path", model_path)
                self.run.log("config_path", config_path)
                self.run.log("checkpoint_path", checkpoint_path)
            
            return config_path, checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to download Azure ML model: {e}")
            raise

    def load_model(self):
        """Load model with Azure ML integration"""
        try:
            # Download from Azure ML if requested
            if self.use_azure_model:
                self.config_filename, self.checkpoint_filename = self.download_azure_model_files()
            
            # Load configuration
            self.config = mmcv.Config.fromfile(self.config_filename)
            self.config.model.pretrained = None
            self.config.model.train_cfg = None
            
            logger.info(f"Loaded config from: {self.config_filename}")
            
            # Determine device (prefer GPU if available)
            if torch.cuda.is_available():
                device = "cuda:0"
                logger.info("Using GPU for inference")
            else:
                device = "cpu"
                logger.info("Using CPU for inference")
            
            # Initialize model
            if self.checkpoint_filename is not None:
                self.model = init_segmentor(self.config, self.checkpoint_filename, device=device)
                self.checkpoint = load_checkpoint(
                    self.model, self.checkpoint_filename, map_location=device
                )
                
                # Set model metadata
                if "meta" in self.checkpoint:
                    self.model.CLASSES = self.checkpoint["meta"].get("CLASSES", [])
                    self.model.PALETTE = self.checkpoint["meta"].get("PALETTE", [])
                    
                    logger.info(f"Model classes: {len(self.model.CLASSES)}")
                    
                    if self.run:
                        self.run.log("num_classes", len(self.model.CLASSES))
                        self.run.log("model_classes", json.dumps(self.model.CLASSES))
            
            self.model.cfg = self.config
            self.model.to(device)
            self.model.eval()
            self.device = next(self.model.parameters()).device
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
            if self.run:
                self.run.log("model_device", str(self.device))
                self.run.log("model_loaded", True)
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.run:
                self.run.log("model_load_error", str(e))
            raise

    def infer(self, images: List[str]) -> List[np.ndarray]:
        """
        Infer on provided images with Azure ML logging
        
        Args:
            images (List[str]): List of image paths
            
        Returns:
            List[np.ndarray]: Inference results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Starting inference on {len(images)} images")
        
        if self.run:
            self.run.log("inference_batch_size", len(images))
        
        try:
            # Prepare test pipeline
            test_pipeline = self.config.data.test.pipeline
            test_pipeline = Compose(test_pipeline)
            
            # Prepare data
            data = []
            for image in images:
                image_data = dict(img_info=dict(filename=image))
                image_data['seg_fields'] = []
                image_data['img_prefix'] = DOWNLOAD_FOLDER
                image_data['seg_prefix'] = DOWNLOAD_FOLDER
                image_data = test_pipeline(image_data)
                data.append(image_data)
            
            # Collate and scatter data
            data = collate(data, samples_per_gpu=len(images))
            
            if next(self.model.parameters()).is_cuda:
                data = scatter(data, [self.device])[0]
            else:
                data["img_metas"] = [i.data[0] for i in list(data["img_metas"])]
            
            # Run inference
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)
            
            logger.info("Inference completed successfully")
            
            if self.run:
                self.run.log("inference_completed", True)
                self.run.log("results_count", len(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            if self.run:
                self.run.log("inference_error", str(e))
            raise

    def infer_single(self, image_path: str) -> np.ndarray:
        """
        Infer on a single image
        
        Args:
            image_path (str): Path to image
            
        Returns:
            np.ndarray: Inference result
        """
        results = self.infer([image_path])
        return results[0] if results else None

    def show_result_pyplot(self,
                          img: Union[str, np.ndarray],
                          result: np.ndarray,
                          palette: Optional[List[List[int]]] = None,
                          fig_size: Tuple[int, int] = (15, 10),
                          opacity: float = 0.5,
                          title: str = "",
                          block: bool = True,
                          out_file: Optional[str] = None,
                          upload_to_azure: bool = False) -> Optional[str]:
        """
        Visualize segmentation results with Azure ML integration
        
        Args:
            img: Image filename or loaded image
            result: The segmentation result
            palette: The palette of segmentation map
            fig_size: Figure size of the pyplot figure
            opacity: Opacity of painted segmentation map
            title: The title of pyplot figure
            block: Whether to block the pyplot figure
            out_file: The path to write the image
            upload_to_azure: Whether to log image to Azure ML
            
        Returns:
            str: Path to saved image if out_file specified
        """
        try:
            if hasattr(self.model, "module"):
                model = self.model.module
            else:
                model = self.model
            
            # Generate visualization
            img_result = model.show_result(
                img, result, 
                palette=palette, 
                show=False, 
                opacity=opacity
            )
            
            # Create matplotlib figure
            plt.figure(figsize=fig_size)
            plt.imshow(mmcv.bgr2rgb(img_result))
            plt.title(title)
            plt.tight_layout()
            
            # Save to file if specified
            if out_file is not None:
                mmcv.imwrite(img_result, out_file)
                logger.info(f"Visualization saved to: {out_file}")
                
                # Log to Azure ML if available
                if self.run and upload_to_azure:
                    try:
                        self.run.log_image("segmentation_result", path=out_file)
                        logger.info("Image logged to Azure ML")
                    except Exception as e:
                        logger.warning(f"Failed to log image to Azure ML: {e}")
            
            plt.show(block=block)
            return out_file
            
        except Exception as e:
            logger.error(f"Failed to show result: {e}")
            if self.run:
                self.run.log("visualization_error", str(e))
            raise

    def batch_infer_and_save(self, 
                           images: List[str], 
                           output_dir: str,
                           save_visualizations: bool = True,
                           upload_to_azure: bool = False) -> List[str]:
        """
        Batch inference with automatic saving and Azure ML logging
        
        Args:
            images: List of image paths
            output_dir: Directory to save results
            save_visualizations: Whether to save visualization images
            upload_to_azure: Whether to upload results to Azure ML
            
        Returns:
            List[str]: List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        logger.info(f"Starting batch inference on {len(images)} images")
        
        # Run inference
        results = self.infer(images)
        
        for i, (image_path, result) in enumerate(zip(images, results)):
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save raw result
                result_file = os.path.join(output_dir, f"{base_name}_result.npy")
                np.save(result_file, result)
                output_files.append(result_file)
                
                # Save visualization if requested
                if save_visualizations:
                    viz_file = os.path.join(output_dir, f"{base_name}_visualization.png")
                    self.show_result_pyplot(
                        img=image_path,
                        result=result,
                        title=f"Segmentation: {base_name}",
                        block=False,
                        out_file=viz_file,
                        upload_to_azure=upload_to_azure
                    )
                    output_files.append(viz_file)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(images)} images")
                    if self.run:
                        progress = ((i + 1) / len(images)) * 100
                        self.run.log("batch_inference_progress", progress)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                if self.run:
                    self.run.log("processing_error", str(e))
        
        logger.info(f"Batch inference completed. {len(output_files)} files created.")
        
        if self.run:
            self.run.log("batch_inference_completed", True)
            self.run.log("total_output_files", len(output_files))
        
        return output_files

    def get_model_info(self) -> dict:
        """Get model information for logging/debugging"""
        if not self.model:
            return {}
        
        info = {
            "config_file": self.config_filename,
            "checkpoint_file": self.checkpoint_filename,
            "device": str(self.device),
            "classes": getattr(self.model, 'CLASSES', []),
            "num_classes": len(getattr(self.model, 'CLASSES', [])),
            "model_type": type(self.model).__name__
        }
        
        if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'model'):
            info["model_config"] = self.model.cfg.model
        
        return info

    def log_model_info(self):
        """Log model information to Azure ML"""
        if not self.run:
            return
        
        info = self.get_model_info()
        for key, value in info.items():
            if isinstance(value, (str, int, float, bool)):
                self.run.log(f"model_{key}", value)
            else:
                self.run.log(f"model_{key}", str(value))
