#!/bin/bash
# DGX Spark PyTorch Development Environment
# Usage: ./start_pytorch.sh [command]
#   ./start_pytorch.sh           # Start Jupyter Lab
#   ./start_pytorch.sh bash      # Get shell
#   ./start_pytorch.sh python    # Python REPL

IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"
CONTAINER_NAME="dgx-spark-pytorch"

# Default command is Jupyter Lab
CMD="${@:-jupyter lab --ip=0.0.0.0 --allow-root --no-browser}"

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "Starting DGX Spark PyTorch environment..."
echo "Image: $IMAGE"
echo "Command: $CMD"
echo ""

docker run --gpus all -it --rm \
    --name $CONTAINER_NAME \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.cache/torch:/root/.cache/torch \
    -p 8888:8888 \
    -p 6006:6006 \
    --ipc=host \
    -w /workspace \
    $IMAGE \
    $CMD
