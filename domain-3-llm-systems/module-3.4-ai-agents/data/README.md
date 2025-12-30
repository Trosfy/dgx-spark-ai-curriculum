# Module 13 Data Directory

This directory contains sample documents and data for the AI Agents module.

## Directory Structure

```
data/
├── sample_documents/          # Sample technical documents for RAG
│   ├── dgx_spark_overview.txt
│   ├── transformer_architecture.txt
│   ├── lora_finetuning.txt
│   ├── quantization_guide.txt
│   └── deployment_best_practices.txt
├── evaluation/                # Evaluation datasets for agents
│   └── agent_test_cases.json
└── README.md
```

## Sample Documents

The `sample_documents/` folder contains technical documentation about AI/ML topics.
These documents are used for building and testing RAG (Retrieval-Augmented Generation) systems.

### Document Topics:

| Document | Size | Description |
|----------|------|-------------|
| `dgx_spark_overview.txt` | 4.2 KB | Hardware specs and capabilities of the DGX Spark system |
| `transformer_architecture.txt` | 4.9 KB | Deep dive into transformer neural networks |
| `lora_finetuning.txt` | 6.0 KB | Low-Rank Adaptation for efficient model fine-tuning |
| `quantization_guide.txt` | 6.9 KB | Reducing model size with quantization techniques |
| `deployment_best_practices.txt` | 9.8 KB | Production deployment strategies |

**Total size:** ~32 KB

## Evaluation Data

The `evaluation/` folder contains test cases for benchmarking agent performance.

### Test Case Format:
```json
{
    "id": "test_001",
    "query": "What is the memory capacity of DGX Spark?",
    "expected_answer": "128GB unified memory",
    "category": "factual_retrieval",
    "difficulty": "easy"
}
```

## Usage

```python
from pathlib import Path

# Load all documents
data_dir = Path("data/sample_documents")
documents = []
for doc_path in data_dir.glob("*.txt"):
    with open(doc_path, 'r') as f:
        documents.append({
            "content": f.read(),
            "source": doc_path.name
        })
```

## Notes

- All sample documents are educational content created for this curriculum
- Documents are formatted for optimal chunking with ~512 token chunks
- Each document includes clear section headers for better retrieval
