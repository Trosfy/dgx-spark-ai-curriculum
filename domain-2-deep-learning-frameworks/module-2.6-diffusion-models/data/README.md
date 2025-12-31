# Module 2.6: Diffusion Models - Data

This directory contains sample data and documentation for the diffusion models module.

## Directory Structure

```
data/
├── README.md                 # This file
├── sample_prompts.txt        # Example prompts for generation
└── style_images/             # Sample images for LoRA training (optional)
    ├── image_001.jpg
    ├── image_001.txt         # Caption file
    └── ...
```

## Sample Prompts

The `sample_prompts.txt` file contains curated prompts for different styles:

### Photorealistic
```
A professional headshot of a young woman, studio lighting, 8K, sharp focus
A serene mountain landscape at golden hour, national geographic style
Street photography of Tokyo at night, neon lights, rain reflections
```

### Artistic
```
A dreamlike forest with bioluminescent trees, fantasy art style
Abstract geometric patterns, vibrant colors, digital art
A steampunk cityscape, intricate machinery, oil painting style
```

### Characters
```
A wise wizard with a long beard, holding a glowing staff, fantasy portrait
A cyberpunk hacker with neon hair, futuristic goggles, sci-fi style
A cute cartoon cat astronaut floating in space, Pixar style
```

## Datasets for Training

For LoRA training, you'll need 10-50 images in a consistent style. Here are recommended sources:

### Option 1: Personal Images
- Use your own artwork or photographs
- Ensure consistent style across images
- Create `.txt` caption files for each image

### Option 2: Public Domain
- [Wikimedia Commons](https://commons.wikimedia.org)
- [Unsplash](https://unsplash.com) (check license)
- [Pexels](https://pexels.com) (check license)

### Option 3: Generated Datasets
- Use an existing model to generate training data
- Curate and filter for quality/consistency
- Good for style transfer experiments

## Caption File Format

Each image should have a corresponding `.txt` file with the same name:

```
# For image "sunset_001.jpg", create "sunset_001.txt" containing:
A vibrant sunset over the ocean, golden and orange hues,
dramatic clouds, professional landscape photography
```

### Caption Best Practices

1. **Be descriptive**: Include subject, setting, style, colors
2. **Use trigger words**: Add a unique word like `sks` for DreamBooth
3. **Include style tokens**: "oil painting", "digital art", "photograph"
4. **Mention quality**: "high detail", "sharp focus", "8K"

Example caption structure:
```
[subject], [setting/background], [style], [quality modifiers]
```

## Memory Considerations

| Dataset Size | Estimated Memory | Training Time (estimate) |
|-------------|------------------|-------------------------|
| 10 images   | ~8GB            | 30 minutes             |
| 50 images   | ~12GB           | 2 hours                |
| 200 images  | ~20GB           | 8 hours                |

DGX Spark's 128GB unified memory can handle even large datasets comfortably!

## Data Preparation Script

Use this script to prepare your dataset:

```python
from pathlib import Path
from PIL import Image
import os

def prepare_training_data(
    source_dir: str,
    output_dir: str,
    resolution: int = 1024,
    default_caption: str = "a photo in the style of sks"
):
    """
    Prepare images for LoRA training.

    - Resizes images to target resolution
    - Creates placeholder caption files if missing
    """
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    for img_path in source.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Resize shortest side to resolution
        if w < h:
            new_w = resolution
            new_h = int(h * resolution / w)
        else:
            new_h = resolution
            new_w = int(w * resolution / h)

        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to square
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))

        # Save processed image
        output_path = output / f"{img_path.stem}.jpg"
        img.save(output_path, quality=95)

        # Create caption file if not exists
        caption_src = img_path.with_suffix('.txt')
        caption_dst = output / f"{img_path.stem}.txt"

        if caption_src.exists():
            with open(caption_src) as f:
                caption = f.read()
        else:
            caption = default_caption

        with open(caption_dst, 'w') as f:
            f.write(caption)

        print(f"Processed: {img_path.name}")

# Usage:
# prepare_training_data("./my_images", "./training_data", resolution=1024)
```

## Next Steps

After preparing your data:

1. Check images are correctly sized: `ls -la training_data/`
2. Verify captions exist: `ls *.txt | wc -l`
3. Proceed to Lab 2.6.5: LoRA Style Training
