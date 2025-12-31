# Module 3.4: Test-Time Compute & Reasoning

**Domain:** 3 - LLM Systems
**Duration:** Week 23 (8-10 hours)
**Prerequisites:** Module 3.3 (Deployment & Inference)
**Priority:** P1 High

---

## Overview

The era of reasoning models has arrived. Models like DeepSeek-R1 and OpenAI's o1 demonstrate that spending more compute at inference time—rather than just training time—can dramatically improve performance on complex tasks. This module teaches you to understand, use, and implement test-time compute strategies on DGX Spark.

**ELI5:** Imagine solving a math problem. You could try to answer immediately (fast but risky), or you could show your work step-by-step, check your reasoning, and maybe try a few approaches before picking the best answer. Reasoning models do exactly this—they "think out loud" before responding.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Explain test-time compute scaling and inference-time reasoning
- ✅ Implement Chain-of-Thought and reasoning strategies
- ✅ Use reasoning models (DeepSeek-R1) effectively on DGX Spark
- ✅ Apply Best-of-N sampling with reward models

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.4.1 | Explain test-time compute vs training-time compute tradeoffs | Understand |
| 3.4.2 | Implement Chain-of-Thought and self-consistency prompting | Apply |
| 3.4.3 | Run and evaluate reasoning models (DeepSeek-R1) | Apply |
| 3.4.4 | Implement Best-of-N sampling with reward models | Apply |

---

## Topics

### 3.4.1 Test-Time Compute Scaling

- **Training vs Inference Compute**
  - Traditional scaling: More training compute → better models
  - New paradigm: More inference compute → better answers
  - O1/O3 reasoning approach

- **When to Invest in Test-Time Compute**
  - Complex reasoning tasks (math, code, analysis)
  - High-stakes decisions where accuracy matters
  - Tasks with verifiable answers

- **Cost-Quality Tradeoffs**
  - More tokens = more cost, but better accuracy
  - Finding the right balance for your use case

### 3.4.2 Reasoning Strategies

- **Chain-of-Thought (CoT)**
  - "Let's think step by step"
  - Explicit reasoning traces
  - Zero-shot vs few-shot CoT

- **Self-Consistency**
  - Generate multiple reasoning paths
  - Majority voting on final answer
  - Temperature > 0 for diversity

- **Tree-of-Thought**
  - Explore multiple branches of reasoning
  - Backtrack when paths lead nowhere
  - Search algorithms for reasoning

- **"Let's Verify Step by Step"**
  - Process reward models for step verification
  - Catch errors mid-reasoning

### 3.4.3 Reasoning Models

- **DeepSeek-R1**
  - GRPO training approach
  - Explicit `<think>` tokens
  - Distilled versions: 1.5B, 7B, 14B, 32B, 70B
  - Running on DGX Spark

- **Other Reasoning Models**
  - QwQ from Alibaba
  - Skywork-R1
  - Comparative strengths

### 3.4.4 Reward Models and Selection

- **Reward Model Basics**
  - Scoring model outputs for quality
  - Outcome vs process reward models

- **Best-of-N Sampling**
  - Generate N candidates
  - Score with reward model
  - Select highest-scoring output

- **Process Reward Models (PRM)**
  - Score each reasoning step
  - Guide search toward good paths

---

## Labs

### Lab 3.4.1: Chain-of-Thought Workshop
**Time:** 2 hours

Master CoT prompting for complex reasoning tasks.

**Instructions:**
1. Open `labs/lab-3.4.1-cot-workshop.ipynb`
2. Implement zero-shot CoT with "Let's think step by step"
3. Test on GSM8K math problems (use sample set)
4. Compare accuracy: direct answer vs CoT
5. Try few-shot CoT with hand-crafted examples
6. Document prompting patterns that work best

**Deliverable:** CoT prompting notebook with accuracy measurements

---

### Lab 3.4.2: Self-Consistency Implementation
**Time:** 1.5 hours

Implement majority voting for improved reasoning.

**Instructions:**
1. Open `labs/lab-3.4.2-self-consistency.ipynb`
2. Generate 5 reasoning paths with temperature=0.7
3. Extract final answers from each path
4. Implement majority voting
5. Compare accuracy: single sample vs self-consistency
6. Experiment with different N values (3, 5, 10)

**Deliverable:** Self-consistency implementation with accuracy comparison

---

### Lab 3.4.3: DeepSeek-R1 Exploration
**Time:** 2 hours

Run the state-of-the-art reasoning model on DGX Spark.

**Instructions:**
1. Open `labs/lab-3.4.3-deepseek-r1.ipynb`
2. Load R1-distill-70B via Ollama (Q4 quantization)
3. Test on math problems—observe the `<think>` process
4. Test on coding challenges
5. Test on complex analysis tasks
6. Document the thinking patterns you observe

**Deliverable:** R1 exploration notebook with thinking analysis

---

### Lab 3.4.4: R1 vs Standard Model Comparison
**Time:** 1.5 hours

Quantify the reasoning advantage.

**Instructions:**
1. Open `labs/lab-3.4.4-r1-comparison.ipynb`
2. Create test set: 20 math, 10 code, 10 reasoning problems
3. Run Llama 3.1 70B on all problems
4. Run DeepSeek-R1-distill-70B on all problems
5. Score and compare accuracy
6. Analyze token usage: thinking overhead worth it?

**Deliverable:** Comparison report with accuracy and cost analysis

---

### Lab 3.4.5: Best-of-N with Reward Model
**Time:** 2 hours

Implement reward-guided generation.

**Instructions:**
1. Open `labs/lab-3.4.5-best-of-n.ipynb`
2. Load a reward model (e.g., ArmoRM or Skywork-Reward)
3. Generate N=5 candidates for each prompt
4. Score each with reward model
5. Select best candidate
6. Compare quality vs greedy decoding

**Deliverable:** Best-of-N implementation with quality measurements

---

### Lab 3.4.6: Reasoning Pipeline
**Time:** 2 hours

Build an intelligent routing system.

**Instructions:**
1. Open `labs/lab-3.4.6-reasoning-pipeline.ipynb`
2. Implement complexity classifier (simple vs complex queries)
3. Route simple queries to fast model (8B)
4. Route complex queries to reasoning model (R1-70B)
5. Add caching for repeated reasoning patterns
6. Measure overall latency and quality

**Deliverable:** Adaptive reasoning pipeline with routing logic

---

## Guidance

### Running DeepSeek-R1 on DGX Spark

```bash
# Pull the distilled 70B model
ollama pull deepseek-r1:70b

# Or use Q4 quantization for 128GB system
ollama run deepseek-r1:70b-q4_K_M
```

```python
# Example: Observing R1's thinking process
import ollama

response = ollama.chat(
    model="deepseek-r1:70b",
    messages=[{
        "role": "user",
        "content": "What is 17 * 23 + 156 / 12?"
    }]
)

# The response includes <think>...</think> blocks
# showing the model's reasoning process
print(response['message']['content'])
```

### Chain-of-Thought Prompting

```python
# Zero-shot CoT
prompt = """
Solve this problem step by step.

Problem: A store sells apples for $2 each and oranges for $3 each.
If I buy 5 apples and some oranges and spend $25 total, how many oranges did I buy?

Let's think step by step:
"""

# Few-shot CoT
few_shot_prompt = """
Q: If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?
A: Let's think step by step.
1. Start with 3 cars
2. 2 more cars arrive
3. Total = 3 + 2 = 5 cars
The answer is 5.

Q: If there are 5 apples and I eat 2, how many are left?
A: Let's think step by step.
1. Start with 5 apples
2. Eat 2 apples
3. Remaining = 5 - 2 = 3 apples
The answer is 3.

Q: {your_question}
A: Let's think step by step.
"""
```

### Self-Consistency Implementation

```python
import collections

def self_consistency(model, prompt, n_samples=5, temperature=0.7):
    """Generate multiple reasoning paths and vote on the answer."""
    answers = []

    for _ in range(n_samples):
        response = model.generate(
            prompt + "\nLet's think step by step:\n",
            temperature=temperature
        )

        # Extract final answer (implementation depends on format)
        answer = extract_answer(response)
        answers.append(answer)

    # Majority vote
    counter = collections.Counter(answers)
    best_answer, count = counter.most_common(1)[0]
    confidence = count / n_samples

    return best_answer, confidence, answers
```

### Best-of-N with Reward Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def best_of_n(llm, prompt, n=5, temperature=0.7):
    """Generate N candidates and select best by reward score."""
    candidates = []

    for _ in range(n):
        response = llm.generate(prompt, temperature=temperature)
        candidates.append(response)

    # Score each candidate
    scores = []
    for candidate in candidates:
        score = reward_model.score(prompt, candidate)
        scores.append(score)

    # Return best candidate
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]
```

### DGX Spark Performance

| Model | Quantization | Memory | Thinking Speed |
|-------|--------------|--------|----------------|
| R1-distill-7B | FP16 | ~14GB | Fast (~50 tok/s) |
| R1-distill-32B | Q4 | ~20GB | Moderate (~30 tok/s) |
| R1-distill-70B | Q4 | ~45GB | Good (~20 tok/s) |
| Llama 3.1 70B | Q4 | ~45GB | Fast (~25 tok/s) |

**Note:** R1 models produce more tokens due to thinking, but the quality improvement on reasoning tasks is substantial.

---

## DGX Spark Setup

### NGC Container Launch

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Install Dependencies

```bash
# Inside NGC container
pip install ollama openai transformers sentence-transformers
```

### Ollama Setup for R1

```bash
# In a separate terminal
ollama serve

# Pull reasoning models
ollama pull deepseek-r1:7b           # Fast testing
ollama pull deepseek-r1:70b          # Best quality
ollama pull llama3.1:70b             # For comparison
```

---

## Milestone Checklist

- [ ] Chain-of-Thought prompting with measured accuracy improvement
- [ ] Self-consistency implementation working with majority voting
- [ ] DeepSeek-R1 running on DGX Spark
- [ ] R1 vs standard model comparison complete
- [ ] Best-of-N sampling with reward model implemented
- [ ] Adaptive reasoning pipeline built

---

## Common Issues

| Issue | Solution |
|-------|----------|
| R1 model too slow | Use smaller distilled version (7B, 14B) or higher quantization |
| Thinking tokens too long | Set max_tokens limit or use shorter problems for testing |
| Self-consistency disagreement | Increase N or use more specific prompts |
| Reward model OOM | Use smaller reward model or batch scoring |
| Ollama connection refused | Ensure `ollama serve` is running in separate terminal |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save your reasoning pipelines to `scripts/`
3. ➡️ Proceed to [Module 3.5: RAG Systems & Vector Databases](../module-3.5-rag-systems/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 3.3: Deployment & Inference](../module-3.3-deployment/) | **Module 3.4: Test-Time Compute** | [Module 3.5: RAG Systems](../module-3.5-rag-systems/) |

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Chain-of-Thought in 5 minutes |
| [ELI5.md](./ELI5.md) | Test-time compute, CoT, reasoning models explained |
| [PREREQUISITES.md](./PREREQUISITES.md) | Self-check before starting |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and lab roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Prompting patterns and code snippets |
| [LAB_PREP.md](./LAB_PREP.md) | Ollama setup and R1 model downloads |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Reasoning model errors and fixes |
| [FAQ.md](./FAQ.md) | Frequently asked questions |

---

## Resources

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2401.02954) - GRPO and reasoning training
- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903) - Original CoT work
- [Self-Consistency Paper](https://arxiv.org/abs/2203.11171) - Majority voting for reasoning
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Process reward models
- [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) - OpenAI's approach
- [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) - Reward model
