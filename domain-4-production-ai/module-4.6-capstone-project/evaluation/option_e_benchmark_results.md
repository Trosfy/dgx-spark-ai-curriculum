# Option E: Browser LLM Benchmark Results Template

## Model Information

| Property | Value |
|----------|-------|
| Base Model | [e.g., Gemma 3 270M Instruct] |
| Fine-tuned Model | [Your model name] |
| Domain | [e.g., Matcha Tea Expert] |
| Training Examples | [X] |
| Final Model Size | [X] MB (INT4 ONNX) |

---

## Training Metrics

### Loss Curves

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | [X.XXX] | [X.XXX] |
| 2 | [X.XXX] | [X.XXX] |
| 3 | [X.XXX] | [X.XXX] |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | [X] |
| Batch Size | [X] |
| Gradient Accumulation | [X] |
| LoRA Rank | [X] |
| LoRA Alpha | [X] |
| Training Time | [X] hours |
| Peak Memory | [X] GB |

---

## Model Size Comparison

| Format | Size | Reduction |
|--------|------|-----------|
| Base Model (BF16) | ~540 MB | - |
| Merged Model (BF16) | ~540 MB | 0% |
| ONNX (FP32) | ~1 GB | +85% |
| ONNX (INT4) | ~150-200 MB | -70% |

---

## Browser Performance Benchmarks

### Desktop Performance

| Device | GPU | Browser | Backend | Load Time | Tokens/sec |
|--------|-----|---------|---------|-----------|------------|
| [Device 1] | [GPU] | Chrome | WebGPU | [X]s | [X] |
| [Device 2] | [GPU] | Edge | WebGPU | [X]s | [X] |
| [Device 3] | - | Chrome | WASM | [X]s | [X] |

### Laptop Performance

| Device | GPU | Browser | Backend | Load Time | Tokens/sec |
|--------|-----|---------|---------|-----------|------------|
| [Laptop 1] | [GPU] | Chrome | WebGPU | [X]s | [X] |
| [Laptop 2] | - | Safari | WASM | [X]s | [X] |

### Mobile Performance (Optional)

| Device | Browser | Backend | Load Time | Tokens/sec |
|--------|---------|---------|-----------|------------|
| [Phone 1] | Chrome | WASM | [X]s | [X] |
| [Tablet 1] | Safari | WASM | [X]s | [X] |

---

## Quality Evaluation

### Automated Metrics

| Metric | Base Model | Fine-Tuned | Change |
|--------|------------|------------|--------|
| Perplexity | [X] | [X] | [X%] |
| BLEU (on test set) | [X] | [X] | [X%] |

### Manual Quality Assessment

**Scoring Rubric:**
- **5 (Excellent)**: Accurate, detailed, well-structured response
- **4 (Good)**: Correct with minor omissions
- **3 (Acceptable)**: Mostly correct, some inaccuracies
- **2 (Poor)**: Significant errors or irrelevant content
- **1 (Unacceptable)**: Incorrect or nonsensical

### Sample Evaluations

#### Question 1: [Basic domain question]

| Model | Response Summary | Score |
|-------|------------------|-------|
| Base Model | [Brief summary] | [X/5] |
| Fine-Tuned | [Brief summary] | [X/5] |
| INT4 ONNX | [Brief summary] | [X/5] |

#### Question 2: [Intermediate domain question]

| Model | Response Summary | Score |
|-------|------------------|-------|
| Base Model | [Brief summary] | [X/5] |
| Fine-Tuned | [Brief summary] | [X/5] |
| INT4 ONNX | [Brief summary] | [X/5] |

#### Question 3: [Advanced domain question]

| Model | Response Summary | Score |
|-------|------------------|-------|
| Base Model | [Brief summary] | [X/5] |
| Fine-Tuned | [Brief summary] | [X/5] |
| INT4 ONNX | [Brief summary] | [X/5] |

#### Question 4: [Edge case / tricky question]

| Model | Response Summary | Score |
|-------|------------------|-------|
| Base Model | [Brief summary] | [X/5] |
| Fine-Tuned | [Brief summary] | [X/5] |
| INT4 ONNX | [Brief summary] | [X/5] |

### Average Quality Scores

| Model | Avg Score | Notes |
|-------|-----------|-------|
| Base Model | [X.X/5] | [General observations] |
| Fine-Tuned (BF16) | [X.X/5] | [General observations] |
| INT4 ONNX (Browser) | [X.X/5] | [General observations] |

---

## Quantization Quality Analysis

### Does INT4 preserve quality?

| Aspect | Preserved? | Notes |
|--------|------------|-------|
| Factual Accuracy | [Yes/Partial/No] | [Details] |
| Response Coherence | [Yes/Partial/No] | [Details] |
| Domain Terminology | [Yes/Partial/No] | [Details] |
| Response Length | [Yes/Partial/No] | [Details] |

### Known Limitations

1. [Limitation 1 with example]
2. [Limitation 2 with example]
3. [Limitation 3 with example]

---

## Deployment Metrics

### Static Asset Sizes

| Asset | Size |
|-------|------|
| HTML/CSS/JS Bundle | [X] KB |
| Model Files (total) | [X] MB |
| Total Download | [X] MB |

### CDN/Hosting Performance

| Metric | Value |
|--------|-------|
| First Load (uncached) | [X]s |
| Subsequent Load (cached) | [X]s |
| Time to First Token | [X]s |

---

## Summary

### Strengths
1. [Key strength 1]
2. [Key strength 2]
3. [Key strength 3]

### Areas for Improvement
1. [Area 1]
2. [Area 2]
3. [Area 3]

### Overall Assessment

[2-3 paragraph summary of the model's performance, suitability for browser deployment, and recommendations]

---

## Appendix: Test Questions Used

1. [Full text of test question 1]
2. [Full text of test question 2]
3. [Full text of test question 3]
4. [Full text of test question 4]
5. [Additional test questions...]
