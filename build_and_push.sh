#!/bin/bash

# Simple Azure Container Registry build and push script
# Update these variables:
ACR_NAME="your-acr-name"
IMAGE_NAME="azureml-hls"
TAG="latest"

# Build and push
docker build . --platform linux/amd64 -t ${IMAGE_NAME}:${TAG}
az acr login --name ${ACR_NAME}
docker tag ${IMAGE_NAME}:${TAG} ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}
docker push ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}

echo "✅ Image pushed: ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${TAG}"
