import os
from os import path
from glob import glob
from azureml.core import Run, Model, Workspace
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Get Azure ML context
try:
    run = Run.get_context()
    ws = run.experiment.workspace
except:
    # For local development/testing
    ws = Workspace.from_config()
    run = None

# Azure Blob Storage configuration
STORAGE_ACCOUNT_URL = os.environ.get('AZURE_STORAGE_ACCOUNT_URL', '')
CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME', 'models')
MODEL_PATH = "models/{model_name}"


def get_blob_service_client():
    """Get Azure Blob Service Client using managed identity or connection string"""
    try:
        # Try using managed identity first (recommended for production)
        credential = DefaultAzureCredential()
        return BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
    except:
        # Fallback to connection string if available
        connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        if connection_string:
            return BlobServiceClient.from_connection_string(connection_string)
        else:
            raise Exception("No valid Azure Storage credentials found")


def save_model_artifacts(model_artifacts_path):
    """
    Save model artifacts to Azure ML and optionally to Azure Blob Storage
    
    Args:
        model_artifacts_path (str): Path to the directory containing model files
    """
    if not path.exists(model_artifacts_path):
        print(f"Model artifacts path does not exist: {model_artifacts_path}")
        return
    
    model_files = glob(f"{model_artifacts_path}/*.pth")
    
    if not model_files:
        print(f"No .pth files found in {model_artifacts_path}")
        return
    
    for model_file in model_files:
        model_name = model_file.split('/')[-1]
        model_name = os.environ.get('MODEL_NAME', model_name)
        
        # Method 1: Register model with Azure ML (Recommended)
        if run:
            # During training run - upload to run outputs
            run.upload_file(name=f"outputs/{model_name}", path_or_stream=model_file)
            print(f"Uploaded model to Azure ML run outputs: {model_name}")
            
            # Register the model
            model = run.register_model(
                model_name=model_name.replace('.pth', ''),
                model_path=f"outputs/{model_name}",
                description=f"Model trained in run {run.id}",
                tags={'framework': 'pytorch', 'type': 'classification'}
            )
            print(f"Registered model: {model.name} version {model.version}")
        else:
            # Local development - register directly to workspace
            model = Model.register(
                workspace=ws,
                model_name=model_name.replace('.pth', ''),
                model_path=model_file,
                description="Model registered locally",
                tags={'framework': 'pytorch', 'type': 'classification'}
            )
            print(f"Registered model: {model.name} version {model.version}")
        
        # Method 2: Also save to Azure Blob Storage (Optional)
        try:
            blob_service_client = get_blob_service_client()
            blob_name = MODEL_PATH.format(model_name=model_name)
            
            with open(model_file, "rb") as data:
                blob_service_client.get_blob_client(
                    container=CONTAINER_NAME, 
                    blob=blob_name
                ).upload_blob(data, overwrite=True)
            
            print(f"Uploaded model to Azure Blob Storage: {CONTAINER_NAME}/{blob_name}")
            
        except Exception as e:
            print(f"Warning: Could not upload to blob storage: {str(e)}")


def save_model_to_datastore(model_artifacts_path, datastore_name='workspaceblobstore'):
    """
    Alternative method: Save model artifacts to Azure ML Datastore
    
    Args:
        model_artifacts_path (str): Path to the directory containing model files
        datastore_name (str): Name of the Azure ML datastore
    """
    if not path.exists(model_artifacts_path):
        print(f"Model artifacts path does not exist: {model_artifacts_path}")
        return
    
    try:
        datastore = ws.datastores[datastore_name]
        
        for model_file in glob(f"{model_artifacts_path}/*.pth"):
            model_name = model_file.split('/')[-1]
            model_name = os.environ.get('MODEL_NAME', model_name)
            target_path = MODEL_PATH.format(model_name=model_name)
            
            datastore.upload_files(
                files=[model_file],
                target_path=target_path,
                overwrite=True
            )
            print(f"Uploaded model to datastore: {datastore_name}/{target_path}")
            
    except Exception as e:
        print(f"Error uploading to datastore: {str(e)}")


def download_model_from_registry(model_name, version=None, download_path="./models"):
    """
    Download model from Azure ML Model Registry
    
    Args:
        model_name (str): Name of the registered model
        version (int, optional): Specific version to download. If None, gets latest
        download_path (str): Local path to download the model
    
    Returns:
        str: Path to downloaded model file
    """
    try:
        if version:
            model = Model(ws, name=model_name, version=version)
        else:
            model = Model(ws, name=model_name)
        
        model_path = model.download(target_dir=download_path, exist_ok=True)
        print(f"Downloaded model {model_name} v{model.version} to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None


def print_files_in_path(path_to_check):
    """Print all files in the given path recursively"""
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path_to_check):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)


def log_metrics(metrics_dict):
    """
    Log metrics to Azure ML run
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
    """
    if run:
        for metric_name, metric_value in metrics_dict.items():
            run.log(metric_name, metric_value)
        print(f"Logged metrics: {list(metrics_dict.keys())}")
    else:
        print(f"No active run context. Metrics: {metrics_dict}")


def log_artifact(file_path, artifact_name=None):
    """
    Log artifact to Azure ML run
    
    Args:
        file_path (str): Path to the file to log
        artifact_name (str, optional): Name for the artifact. If None, uses filename
    """
    if run and path.exists(file_path):
        name = artifact_name or path.basename(file_path)
        run.upload_file(name=f"outputs/{name}", path_or_stream=file_path)
        print(f"Logged artifact: {name}")
    else:
        print(f"Cannot log artifact: no run context or file not found: {file_path}")
