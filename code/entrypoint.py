# Azure ML compatible entrypoint - converted from AWS SageMaker
import os
import shlex
import subprocess
import sys
import logging
from subprocess import CalledProcessError
from typing import Optional

from retrying import retry

# Configure logging for Azure ML
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _retry_if_error(exception):
    """Determine if we should retry on this exception"""
    return isinstance(exception, (CalledProcessError, OSError))


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_azure_ml_service():
    """Start the Azure ML compatible service"""
    try:
        logger.info("Starting Azure ML service...")
        
        # Set Azure ML specific environment variables
        port = os.environ.get("AZUREML_MODEL_SERVER_PORT", "8080")
        workers = os.environ.get("AZUREML_MODEL_SERVER_WORKERS", "1")
        
        # Start FastAPI server for Azure ML
        cmd = [
            "python", "-m", "uvicorn", 
            "main:app",
            "--host", "0.0.0.0",
            "--port", port,
            "--workers", workers
        ]
        
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
    except Exception as e:
        logger.error(f"Error starting Azure ML service: {str(e)}")
        raise


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_scoring_service():
    """Start Azure ML scoring service (alternative mode)"""
    try:
        logger.info("Starting Azure ML scoring service...")
        
        # For Azure ML managed endpoints, we use the scoring script directly
        from geospatial_fm.infer import init, run
        
        # Initialize the model
        init()
        logger.info("Azure ML scoring service initialized successfully")
        
        # Keep the service alive
        import time
        while True:
            time.sleep(60)
            logger.info("Scoring service heartbeat")
            
    except Exception as e:
        logger.error(f"Error in scoring service: {str(e)}")
        raise


def _start_development_server():
    """Start development server with auto-reload"""
    try:
        logger.info("Starting development server...")
        
        cmd = [
            "python", "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ]
        
        subprocess.check_call(cmd)
        
    except Exception as e:
        logger.error(f"Error starting development server: {str(e)}")
        raise


def main():
    """Main entrypoint for Azure ML deployment"""
    try:
        logger.info(f"Azure ML entrypoint called with args: {sys.argv}")
        
        if len(sys.argv) < 2:
            logger.error("No command provided")
            sys.exit(1)
        
        command = sys.argv[1].lower()
        
        if command == "serve":
            logger.info("Starting Azure ML serve mode")
            _start_azure_ml_service()
            
        elif command == "score":
            logger.info("Starting Azure ML scoring mode")
            _start_scoring_service()
            
        elif command == "dev":
            logger.info("Starting development mode")
            _start_development_server()
            
        elif command == "init":
            logger.info("Running initialization only")
            from geospatial_fm.infer import init
            init()
            logger.info("Initialization complete")
            
        elif command == "health":
            logger.info("Running health check")
            # Simple health check
            try:
                from geospatial_fm.infer import _service
                if _service.initialized:
                    logger.info("Service is healthy")
                    print("HEALTHY")
                else:
                    logger.warning("Service not initialized")
                    print("UNHEALTHY")
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                print("UNHEALTHY")
                sys.exit(1)
                
        else:
            # Execute custom command
            logger.info(f"Executing custom command: {' '.join(sys.argv[1:])}")
            subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in main entrypoint: {str(e)}")
        raise


def azure_ml_init():
    """Azure ML specific initialization function"""
    try:
        logger.info("Azure ML initialization started")
        
        # Import and initialize the inference service
        from geospatial_fm.infer import init
        init()
        
        logger.info("Azure ML initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Azure ML initialization failed: {str(e)}")
        return False


def azure_ml_run(raw_data):
    """Azure ML specific run function"""
    try:
        logger.info("Azure ML run function called")
        
        from geospatial_fm.infer import run
        result = run(raw_data)
        
        logger.info("Azure ML run function completed")
        return result
        
    except Exception as e:
        logger.error(f"Azure ML run function failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
