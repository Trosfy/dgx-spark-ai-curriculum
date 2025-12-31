# Data Directory for Module 2.2: Computer Vision

This directory contains datasets and sample files for the computer vision module.

## Datasets Used

### CIFAR-10 / CIFAR-100
- **Source**: [CIFAR Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size**: ~170MB (CIFAR-10), ~160MB (CIFAR-100)
- **Classes**: 10 (CIFAR-10) or 100 (CIFAR-100)
- **Image Size**: 32×32 RGB
- **Download**: Automatic via `torchvision.datasets.CIFAR10()`

### Pascal VOC 2012
- **Source**: [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- **Size**: ~2GB
- **Classes**: 21 (including background)
- **Used for**: Semantic segmentation
- **Download**: Automatic via `torchvision.datasets.VOCSegmentation()`

### SAM Checkpoints
- **Source**: [Segment Anything](https://segment-anything.com/)
- **Checkpoints**:
  - `sam_vit_b_01ec64.pth` (375 MB)
  - `sam_vit_l_0b3195.pth` (1.2 GB)
  - `sam_vit_h_4b8939.pth` (2.5 GB)
- **Download**: Automatic in notebook 06

## Directory Structure

```
data/
├── README.md              # This file
├── cifar-10-batches-py/   # CIFAR-10 data (auto-downloaded)
├── cifar-100-python/      # CIFAR-100 data (auto-downloaded)
├── VOCdevkit/             # Pascal VOC data (auto-downloaded)
│   └── VOC2012/
│       ├── JPEGImages/    # RGB images
│       ├── SegmentationClass/  # Segmentation masks
│       └── ...
├── detection_samples/     # Sample images for object detection
│   ├── street.jpg
│   └── zidane.jpg
├── sam_samples/           # Sample images for SAM demos
│   ├── dogs.jpg
│   ├── groceries.jpg
│   └── room.jpg
└── sam_checkpoints/       # SAM model checkpoints (auto-downloaded)
    └── sam_vit_b_01ec64.pth
```

## DGX Spark Memory Considerations

With DGX Spark's 128GB unified memory:
- All datasets can be loaded into memory simultaneously
- The largest SAM checkpoint (ViT-H, 2.5GB) loads easily
- Multiple models can be kept in memory for comparison

## Data Download

All datasets are downloaded automatically when you run the notebooks for the first time. Manual download is not required.

### Clearing Data Cache

To free up disk space, you can delete the downloaded data:

```bash
# Remove all downloaded data
rm -rf data/cifar-10-batches-py
rm -rf data/cifar-100-python
rm -rf data/VOCdevkit
rm -rf data/sam_checkpoints
rm -rf data/detection_samples
rm -rf data/sam_samples
```

## VOC Class Color Mapping

| ID | Class          | Color (RGB)       |
|----|----------------|-------------------|
| 0  | background     | (0, 0, 0)         |
| 1  | aeroplane      | (128, 0, 0)       |
| 2  | bicycle        | (0, 128, 0)       |
| 3  | bird           | (128, 128, 0)     |
| 4  | boat           | (0, 0, 128)       |
| 5  | bottle         | (128, 0, 128)     |
| 6  | bus            | (0, 128, 128)     |
| 7  | car            | (128, 128, 128)   |
| 8  | cat            | (64, 0, 0)        |
| 9  | chair          | (192, 0, 0)       |
| 10 | cow            | (64, 128, 0)      |
| 11 | diningtable    | (192, 128, 0)     |
| 12 | dog            | (64, 0, 128)      |
| 13 | horse          | (192, 0, 128)     |
| 14 | motorbike      | (64, 128, 128)    |
| 15 | person         | (192, 128, 128)   |
| 16 | pottedplant    | (0, 64, 0)        |
| 17 | sheep          | (128, 64, 0)      |
| 18 | sofa           | (0, 192, 0)       |
| 19 | train          | (128, 192, 0)     |
| 20 | tvmonitor      | (0, 64, 128)      |

## COCO Classes (for YOLOv8)

YOLOv8 is trained on COCO dataset with 80 classes including:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Additional Resources

- [CIFAR-10 Dataset Paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [Pascal VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Dataset](https://cocodataset.org/)
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)
