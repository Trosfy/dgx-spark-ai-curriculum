# Data Directory

This directory is used for storing datasets during the Module 6 exercises.

## DGX Spark Docker Configuration

When running on DGX Spark, ensure your Docker container is started with:

```bash
docker run --gpus all --ipc=host -it \
    -v $PWD:/workspace -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root
```

> **Critical:** The `--ipc=host` flag is **required** when using `num_workers > 0` in DataLoader.
> Without it, you'll get "unable to open shared memory" errors because PyTorch workers
> use shared memory for inter-process communication.

## Automatic Downloads

The notebooks will automatically download required datasets when run:

| Dataset | Size | Used In | Description |
|---------|------|---------|-------------|
| CIFAR-10 | ~170 MB | All tasks | 60,000 32x32 color images in 10 classes |
| CIFAR-100 | ~170 MB | Task 6.1 | 60,000 32x32 color images in 100 classes |

## Directory Structure After Running Notebooks

```
data/
├── README.md (this file)
├── cifar-10-batches-py/     # CIFAR-10 extracted data
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
└── cifar-10-python.tar.gz   # Downloaded archive
```

## Storage Notes for DGX Spark

- **Total required**: ~500 MB for all datasets
- **Location**: Data is stored locally in this directory
- **Persistence**: Datasets persist between notebook runs
- **Cleanup**: Delete this directory to free space and force re-download

## Custom Datasets

For Task 6.2 (Dataset Pipeline), you can add your own image datasets:

```
data/
└── my_custom_dataset/
    ├── class_a/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_b/
    │   ├── image1.jpg
    │   └── ...
    └── class_c/
        └── ...
```

Then use with the `ImageFolderDataset` class:

```python
from scripts.custom_dataset import ImageFolderDataset, create_transforms

dataset = ImageFolderDataset(
    root_dir='./data/my_custom_dataset',
    transform=create_transforms('train', image_size=224)
)
```

## Checkpoints

Training checkpoints are stored in `./checkpoints/` (relative to notebook location), not in this data directory. Checkpoints are automatically cleaned up at the end of each notebook.

## Troubleshooting

### Download Issues
If downloads fail, try:
1. Check internet connectivity
2. Clear partially downloaded files
3. Manually download from https://www.cs.toronto.edu/~kriz/cifar.html

### Disk Space
If running low on space:
```bash
# Remove downloaded datasets
rm -rf data/cifar-10-batches-py
rm -f data/cifar-10-python.tar.gz
```

### Permission Issues
Ensure write permissions:
```bash
chmod -R u+w data/
```
