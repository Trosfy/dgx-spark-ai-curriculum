# Data for Module C: Mechanistic Interpretability

This module primarily generates data programmatically during the labs, but this directory contains utilities and documentation for datasets used in interpretability research.

## Data Generation Utilities

Most data for this module is generated on-the-fly:

### IOI (Indirect Object Identification) Dataset
```python
from scripts.mech_interp_utils import create_ioi_dataset

# Generate IOI examples
dataset = create_ioi_dataset(n_samples=100)

# Each example contains:
# - clean: "John and Mary went to the store. John gave a book to"
# - corrupted: "John and Mary went to the store. Mary gave a book to"
# - answer: " Mary"
# - wrong_answer: " John"
```

### Induction Dataset
```python
from scripts.mech_interp_utils import create_induction_dataset

# Generate repeated token sequences for testing induction heads
dataset = create_induction_dataset(
    n_samples=100,
    seq_length=50,
    vocab_start=1000,
    vocab_end=10000
)
# Returns: List of token tensors [1, seq_length * 2]
# Pattern: [A][B][C]...[A][B][C]... (repeated)
```

## Pre-trained SAE Resources

For Lab C.4, you can optionally use pre-trained SAEs:

### Community Resources
- **Neuronpedia**: https://www.neuronpedia.org/
  - Interactive explorer for pre-trained SAE features
  - Covers GPT-2, Pythia, and other models

- **SAE Lens Library**: https://github.com/jbloomAus/SAELens
  - Production-quality SAE training code
  - Pre-trained SAEs for various models

### Loading Pre-trained SAEs
```python
# Example: Loading from SAE Lens
from sae_lens import SAE

sae = SAE.load_from_pretrained("gpt2-small-res-jb")
```

## External Datasets

For more comprehensive training, consider:

1. **The Pile** - Diverse text corpus
   - Good for training robust SAEs

2. **OpenWebText** - Web text similar to GPT-2 training
   - Good for analyzing GPT-2 specifically

3. **Code datasets** (e.g., from Hugging Face)
   - Good for finding code-specific features

## Memory Considerations

On DGX Spark (128GB unified memory), you can:
- Cache activations for ~100K+ tokens easily
- Train SAEs with 10K+ features
- Run full patching experiments without OOM

## File Format

When saving activations or features:
```python
import torch

# Save activations
torch.save({
    'activations': activations,  # [n_samples, d_model]
    'layer': layer,
    'prompts': prompts,
    'metadata': {
        'model': 'gpt2-small',
        'timestamp': '2024-01-01'
    }
}, 'activations.pt')

# Load
data = torch.load('activations.pt')
```
