# DGX Spark Ollama Benchmark Results

**Date:** 2026-01-02 16:10:59

**Platform:** NVIDIA DGX Spark (128GB Unified Memory)


## Results

| Model | Prefill (tok/s) | Decode (tok/s) | Memory (GB) |
|-------|-----------------|----------------|-------------|
| qwen3:32b | 182.7 | 9.9 | 0.0 |
| qwen3:8b | 572.1 | 41.9 | 0.0 |
| qwen3:4b | 1171.7 | 76.3 | 0.0 |
| magistral:24b | 2173.2 | 13.9 | 0.0 |
| qwen3-vl:8b | 710.9 | 42.7 | 0.0 |
| rnj-1:8b | 1238.4 | 37.0 | 0.0 |
| qwen3-embedding:4b | 0.0 | 0.0 | 0.0 |
| ministral-3:14b | 11926.7 | 25.4 | 0.0 |
| gpt-oss:20b | 3130.4 | 59.1 | 0.0 |
| devstral-small-2:24b | 6317.5 | 13.7 | 0.0 |
| devstral-2:123b | 1262.6 | 2.6 | 0.0 |
| deepseek-r1:70b | 57.0 | 4.5 | 0.0 |
| nemotron-3-nano:30b | 338.1 | 62.8 | 0.0 |
| deepseek-ocr:3b | 1235.9 | 175.9 | 0.0 |
| gpt-oss:120b | 2212.6 | 42.5 | 0.0 |

## Notes

- Prefill: Prompt processing speed (higher is better)
- Decode: Token generation speed (higher is better)
- Memory: GPU memory used by the model

## Test Configuration

- Prompt: 'Explain the concept of machine learning in simple terms.'
- Max tokens: 100
- Runs per model: 3 (averaged)