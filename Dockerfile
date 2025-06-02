# Azure ML compatible Docker container for ML training
# Based on NVIDIA CUDA base image for GPU support

FROM nvidia/cuda:11.5.2-base-ubuntu20.04

LABEL maintainer="NASA IMPACT"
LABEL description="Azure ML training container for HLS Foundation model"

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.9-dev

RUN apt-get update && apt-get install -y \
    libgl1 \
    python3-pip \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Upgrade pip
RUN python3.9 -m pip install -U pip

# Install core dependencies
RUN pip3 install --no-cache-dir Cython
RUN pip3 install --upgrade pip
RUN pip3 install imagecodecs

# Install PyTorch with CUDA 11.5 support
RUN pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115

# Clone and install HLS Foundation model
RUN git clone https://github.com/nasa-impact/hls-foundation-os.git
RUN cd hls-foundation-os && git checkout 9cdb612 && pip3 install -e .

# Copy and install requirements (now with Azure ML dependencies)
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Install OpenMMLab dependencies
RUN pip3 install -U openmim
RUN mim install mmengine==0.7.4
RUN mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
RUN pip3 install yapf==0.40.1

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0,1,2
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"

# Create necessary directories
RUN mkdir -p /app/models
RUN mkdir -p /app/outputs
RUN mkdir -p /tmp/data

# Azure ML specific environment variables
ENV AZUREML_MODEL_DIR=/app/models
ENV AZUREML_OUTPUT_DIR=/app/outputs

# Copy application code to /app (Azure ML standard location)
COPY train.py /app/train.py
COPY utils.py /app/utils.py

# Set working directory to /app
WORKDIR /app

# Make scripts executable
RUN chmod +x /app/train.py

# Azure ML expects the training script to be runnable with python
# The entry point will be specified in the Azure ML job configuration
ENTRYPOINT ["python3", "train.py"]

# Optional: Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || exit 1
