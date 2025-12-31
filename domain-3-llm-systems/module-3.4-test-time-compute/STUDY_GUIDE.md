# Module 3.4: Test-Time Compute & Reasoning - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Explain** test-time compute scaling and why inference-time reasoning matters
2. **Implement** Chain-of-Thought and self-consistency prompting
3. **Run** and evaluate reasoning models (DeepSeek-R1) on DGX Spark
4. **Apply** Best-of-N sampling with reward models for quality improvement

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.4.1-cot-workshop.ipynb | Chain-of-Thought mastery | ~2 hr | +15% accuracy on reasoning tasks |
| 2 | lab-3.4.2-self-consistency.ipynb | Multiple paths + voting | ~1.5 hr | Confidence-calibrated answers |
| 3 | lab-3.4.3-deepseek-r1.ipynb | Reasoning model exploration | ~2 hr | See `<think>` tokens in action |
| 4 | lab-3.4.4-r1-comparison.ipynb | R1 vs standard models | ~1.5 hr | Quantify reasoning advantage |
| 5 | lab-3.4.5-best-of-n.ipynb | Reward model selection | ~2 hr | Pick best from multiple candidates |
| 6 | lab-3.4.6-reasoning-pipeline.ipynb | Adaptive routing | ~2 hr | Smart routing by complexity |

**Total time**: ~11 hours

## 🔑 Core Concepts

### Test-Time Compute
**What**: Spending more compute during inference (not training) to improve answer quality
**Why it matters**: Enables smarter answers without retraining, crucial for complex tasks
**First appears in**: Lab 3.4.1

### Chain-of-Thought (CoT)
**What**: Prompting the model to reason step-by-step before answering
**Why it matters**: +15-20% accuracy on math and reasoning tasks with just 5 words
**First appears in**: Lab 3.4.1

### Reasoning Models
**What**: Models trained to generate explicit reasoning (like DeepSeek-R1 with `<think>` tokens)
**Why it matters**: State-of-the-art on complex reasoning, runs on DGX Spark
**First appears in**: Lab 3.4.3

### Best-of-N Sampling
**What**: Generate N candidates, score with reward model, return best
**Why it matters**: Quality improvement without model changes
**First appears in**: Lab 3.4.5

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 3.3          ──►     Module 3.4          ──►    Module 3.5
Deployment                  Reasoning                  RAG Systems
(serve models)              (smarter answers)          (retrieval)
```

**Builds on**:
- **Deployed models** from Module 3.3 (Ollama, vLLM)
- **Model understanding** from Module 3.1 (how LLMs generate)
- **Quality evaluation** from Module 3.2 (perplexity, benchmarks)

**Prepares for**:
- Module 3.5 can use reasoning models for RAG quality
- Module 3.6 uses reasoning for agent decision-making
- Module 4.2 covers safety for reasoning systems

## 📖 Recommended Approach

### Standard Path (8-10 hours):
1. **Day 1: Prompting Strategies (Labs 1-2)**
   - Lab 3.4.1 establishes CoT fundamentals
   - Lab 3.4.2 adds self-consistency

2. **Day 2: Reasoning Models (Labs 3-4)**
   - Lab 3.4.3 explores DeepSeek-R1
   - Lab 3.4.4 quantifies improvements

3. **Day 3: Advanced Techniques (Labs 5-6)**
   - Lab 3.4.5 introduces reward models
   - Lab 3.4.6 builds adaptive routing

### Quick Path (5-6 hours, if experienced):
1. Skim Lab 3.4.1, focus on few-shot CoT patterns
2. Do Lab 3.4.3 (DeepSeek-R1) - the highlight
3. Skip to Lab 3.4.5 if you want reward models
4. Lab 3.4.6 for practical routing

## 📋 Before You Start
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup (Ollama, R1 models)
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute CoT demo
→ See [ELI5.md](./ELI5.md) for concept explanations
