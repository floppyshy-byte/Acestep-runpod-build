#!/usr/bin/env bash
set -e

echo "Building ACE-Step 1.5 RunPod Serverless image..."

IMAGE_NAME="ace-step-1.5-serverless"
TAG="latest"

# Build
docker build -t ${IMAGE_NAME}:${TAG} .

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To test locally with GPU:"
echo "  docker run --rm --gpus all -p 8000:8000 ${IMAGE_NAME}:${TAG}"
echo ""
echo "To push to Docker Hub (for RunPod serverless):"
echo "  docker tag ${IMAGE_NAME}:${TAG} yourdockerhub/${IMAGE_NAME}:${TAG}"
echo "  docker push yourdockerhub/${IMAGE_NAME}:${TAG}"
echo ""
echo "Then create a RunPod serverless endpoint with that image."
