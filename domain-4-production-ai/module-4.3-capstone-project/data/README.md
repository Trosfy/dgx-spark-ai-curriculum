# Capstone Data Directory

This directory is for storing data files used in your capstone project.

## Directory Structure

```
data/
├── raw/           # Original, unprocessed data
├── processed/     # Cleaned and preprocessed data
├── evaluation/    # Evaluation datasets
└── README.md      # This file
```

## Data Guidelines

### What to Store

- **Training datasets** - JSONL files with instruction/output pairs
- **Evaluation sets** - Test cases for benchmarking
- **Knowledge bases** - Documents for RAG systems
- **Sample outputs** - Example inputs/outputs for demos

### What NOT to Store

- **Large model files** - Use `~/.cache/huggingface` instead
- **Sensitive data** - No API keys, passwords, or PII
- **Large raw datasets** - Use Hugging Face datasets library

### File Formats

| Type | Format | Example |
|------|--------|---------|
| Training data | JSONL | `{"instruction": "...", "output": "..."}` |
| Evaluation | JSONL | `{"input": "...", "expected": "..."}` |
| Documents | PDF, TXT, MD | Source documents |
| Configs | JSON, YAML | Configuration files |

### Data Versioning

Consider using:
- Git LFS for large files
- DVC (Data Version Control) for datasets
- Hugging Face datasets for public data

### Example Training Data Format

```json
{"instruction": "What is machine learning?", "input": "", "output": "Machine learning is..."}
{"instruction": "Explain this code", "input": "def foo(): pass", "output": "This is a function..."}
```

### Example Evaluation Data Format

```json
{"id": "1", "input": "Capital of France?", "expected": "Paris", "category": "factual"}
{"id": "2", "input": "Explain photosynthesis", "expected": "...", "category": "explanation"}
```

## Storage Limits

- Keep individual files under 100MB
- Total data directory under 1GB (for git)
- Use external storage for larger datasets

## Resources

- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [DVC Documentation](https://dvc.org/doc)
- [Git LFS](https://git-lfs.github.com/)
