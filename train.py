from __future__ import absolute_import

import argparse
import copy
import mmcv
import os
import os.path as osp
import random as rd
import subprocess
import sys
import time
import torch
import torch.distributed as dist
import warnings

from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_device, get_root_logger, setup_multi_processes

# Azure ML imports
from azureml.core import Run, Dataset, Workspace
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

from utils import print_files_in_path, save_model_artifacts, log_metrics, log_artifact

# Azure ML Run context
try:
    run = Run.get_context()
    ws = run.experiment.workspace
    print(f"Running in Azure ML experiment: {run.experiment.name}")
except:
    # For local development/testing
    ws = Workspace.from_config() if os.path.exists('.azureml/config.json') else None
    run = None
    print("Running locally - no Azure ML context")


def get_azure_storage_client():
    """Get Azure Blob Storage client for data access"""
    storage_account_url = os.environ.get('AZURE_STORAGE_ACCOUNT_URL')
    if not storage_account_url:
        raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable is required")
    
    try:
        # Use managed identity (recommended for Azure ML compute)
        credential = DefaultAzureCredential()
        return BlobServiceClient(account_url=storage_account_url, credential=credential)
    except:
        # Fallback to connection string
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if connection_string:
            return BlobServiceClient.from_connection_string(connection_string)
        else:
            raise Exception("No valid Azure Storage credentials found")


def download_data_from_blob(blob_url, split):
    """
    Download data from Azure Blob Storage
    
    Args:
        blob_url (str): Azure blob URL (e.g., https://account.blob.core.windows.net/container/path)
        split (str): Data split name (training, validation, test, etc.)
    """
    split_folder = f"/tmp/data/{split}"  # Changed from /opt/ml/data to /tmp/data
    if not os.path.exists(split_folder):
        os.makedirs(split_folder)
    
    try:
        blob_service_client = get_azure_storage_client()
        
        # Parse blob URL to get container and prefix
        url_parts = blob_url.replace('https://', '').split('/')
        container_name = url_parts[1]
        prefix = '/'.join(url_parts[2:]) + f'/{split}' if len(url_parts) > 2 else split
        
        container_client = blob_service_client.get_container_client(container_name)
        
        print(f"Downloading {split} data from {container_name}/{prefix}")
        
        # List and download blobs with the specified prefix
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blob_list:
            if blob.name.endswith('/'):  # Skip directories
                continue
                
            filename = blob.name.split('/')[-1]
            if filename:  # Only download actual files
                local_path = os.path.join(split_folder, filename)
                
                # Create subdirectories if needed
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob.name
                )
                
                with open(local_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
                print(f"Downloaded: {filename}")
        
        print(f"Finished downloading {split} data.")
        
    except Exception as e:
        print(f"Error downloading data for {split}: {str(e)}")
        raise


def download_data_from_dataset(dataset_name, split):
    """
    Alternative: Download data from Azure ML Dataset
    
    Args:
        dataset_name (str): Name of the registered Azure ML dataset
        split (str): Data split name
    """
    if not ws:
        print("No workspace context available for dataset download")
        return
    
    try:
        dataset = Dataset.get_by_name(ws, f"{dataset_name}_{split}")
        split_folder = f"/tmp/data/{split}"
        
        print(f"Downloading dataset: {dataset_name}_{split}")
        dataset.download(target_path=split_folder, overwrite=True)
        print(f"Downloaded dataset to: {split_folder}")
        
    except Exception as e:
        print(f"Dataset {dataset_name}_{split} not found, trying blob storage method: {str(e)}")
        # Fallback to blob storage method
        blob_url = os.environ.get('AZURE_BLOB_URL', os.environ.get('DATA_URL'))
        if blob_url:
            download_data_from_blob(blob_url, split)


def setup_logging_and_metrics():
    """Setup Azure ML logging integration"""
    if run:
        # Log environment info to Azure ML
        env_info = {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        for key, value in env_info.items():
            run.log(f"env_{key}", str(value))


def train():
    """Main training function"""
    
    # Setup Azure ML logging
    setup_logging_and_metrics()
    
    # Print data channel info (Azure ML equivalent)
    train_path = os.environ.get("AZUREML_DATAREFERENCE_TRAIN", "/tmp/data/training")
    validation_path = os.environ.get("AZUREML_DATAREFERENCE_VALIDATION", "/tmp/data/validation")
    test_path = os.environ.get("AZUREML_DATAREFERENCE_TEST", "/tmp/data/test")
    
    print(f"\nTrain data path: {train_path}")
    print(f"Validation data path: {validation_path}")
    print(f"Test data path: {test_path}")

    config_file = os.environ.get('CONFIG_FILE', '/tmp/data/configs/default_config.py')
    print(f'\nConfig file: {config_file}')

    print(f"Environment variables: {dict(os.environ)}")

    # Download and prepare data for training
    data_source = os.environ.get('AZURE_BLOB_URL', os.environ.get('DATA_URL'))
    dataset_name = os.environ.get('DATASET_NAME')
    
    for split in ['training', 'validation', 'test', 'configs', 'models']:
        try:
            if dataset_name:
                # Try Azure ML Dataset first
                download_data_from_dataset(dataset_name, split)
            elif data_source:
                # Fallback to direct blob storage
                download_data_from_blob(data_source, split)
            else:
                print(f"Warning: No data source specified for {split}")
        except Exception as e:
            print(f"Warning: Could not download {split} data: {str(e)}")

    # Load configuration
    cfg = Config.fromfile(config_file)
    cfg.device = get_device()
    seed = init_random_seed(10, device=cfg.device)
    
    # Initialize logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # Azure ML specific model path - use outputs directory
    model_path = f"./outputs/{os.environ.get('VERSION', 'v1')}/{os.environ.get('EVENT_TYPE', 'training')}/"
    os.makedirs(model_path, exist_ok=True)

    # Update work directory for Azure ML
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./outputs/work_dirs', osp.splitext(osp.basename(config_file))[0])
    
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=True)
    cfg.seed = seed

    # Log seed to Azure ML
    if run:
        run.log('random_seed', seed)

    # Setup distributed training environment
    os.environ['RANK'] = str(torch.cuda.device_count())
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_PORT'] = str(rd.randint(20000, 30000))
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.gpu_ids = range(1)
    distributed = False
    print('#######', cfg.dist_params)

    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_file)))
    
    # Log config file to Azure ML
    if run:
        run.upload_file(name="outputs/config.py", path_or_stream=osp.join(cfg.work_dir, osp.basename(config_file)))

    # Set multi-process settings
    setup_multi_processes(cfg)

    # Initialize metadata
    meta = dict()
    
    # Log environment info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # Log basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta['seed'] = seed
    meta['exp_name'] = osp.basename(config_file)

    # Build model
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Handle SyncBN for non-distributed training
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.'
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    
    # Setup checkpoint configuration
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )
    
    # Add model classes
    model.CLASSES = datasets[0].CLASSES
    meta.update(cfg.checkpoint_config.meta)
    
    # Custom hook for Azure ML logging
    class AzureMLLoggerHook:
        def __init__(self, run_context):
            self.run = run_context
        
        def log_metrics(self, metrics):
            if self.run:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.run.log(key, value)
    
    # Add Azure ML logger hook to config if we have a run context
    if run:
        azure_logger = AzureMLLoggerHook(run)
    
    # Start training
    print("Starting training...")
    train_segmentor(
        model, datasets, cfg, distributed=distributed, validate=True, timestamp=timestamp, meta=meta
    )
    
    print("Training completed. Saving model artifacts...")
    
    # Save model artifacts using Azure ML
    save_model_artifacts(model_path)
    
    # Log final model directory contents
    if run:
        for root, dirs, files in os.walk(cfg.work_dir):
            for file in files:
                if file.endswith(('.pth', '.json', '.log')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, cfg.work_dir)
                    run.upload_file(name=f"outputs/{relative_path}", path_or_stream=file_path)
    
    print("Model artifacts saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters (same as before)
    parser.add_argument("--hp1", type=str)
    parser.add_argument("--hp2", type=int, default=50)
    parser.add_argument("--hp3", type=float, default=0.1)

    # Azure ML data references (equivalent to SageMaker channels)
    parser.add_argument("--train", type=str, default=os.environ.get("AZUREML_DATAREFERENCE_TRAIN", "/tmp/data/training"))
    parser.add_argument("--validation", type=str, default=os.environ.get("AZUREML_DATAREFERENCE_VALIDATION", "/tmp/data/validation"))

    args = parser.parse_args()
    
    # Log hyperparameters to Azure ML
    if run:
        run.log('hp1', args.hp1 or 'None')
        run.log('hp2', args.hp2)
        run.log('hp3', args.hp3)

    train()
