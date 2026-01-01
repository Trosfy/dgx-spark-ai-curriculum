# Module 4.2: AI Safety & Alignment - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement guardrails** using NeMo Guardrails and Llama Guard
2. **Perform red teaming** to find vulnerabilities in LLM applications
3. **Evaluate safety** using TruthfulQA, BBQ, and custom benchmarks
4. **Document responsibly** by creating comprehensive model cards
5. **Apply regulatory knowledge** from EU AI Act and NIST AI RMF

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 0 | 00-ai-safety-overview.ipynb | Safety Landscape | ~1h | Understanding of safety challenges |
| 1 | lab-4.2.1-nemo-guardrails.ipynb | NeMo Guardrails | ~3h | Chatbot with input/output filtering |
| 2 | lab-4.2.2-llama-guard-integration.ipynb | Llama Guard | ~2h | Classification pipeline integrated |
| 3 | lab-4.2.3-automated-red-teaming.ipynb | Red Teaming | ~3h | Vulnerability report with mitigations |
| 4 | lab-4.2.4-safety-benchmark-suite.ipynb | Benchmarks | ~2h | TruthfulQA + BBQ evaluation results |
| 5 | lab-4.2.5-bias-evaluation.ipynb | Bias Evaluation | ~2h | Bias findings with proposed fixes |
| 6 | lab-4.2.6-model-card-creation.ipynb | Model Cards | ~2h | Published model card on HF Hub |

**Total Time**: ~15 hours

---

## Core Concepts

### Prompt Injection
**What**: Attacks where malicious instructions are inserted into prompts to hijack LLM behavior.
**Why it matters**: The #1 vulnerability in LLM applications - ignoring it leads to data leaks, unauthorized actions.
**First appears in**: Lab 4.2.1

### Guardrails
**What**: Systems that filter or modify LLM inputs and outputs to prevent harmful behavior.
**Why it matters**: The primary defense layer between users and your AI system.
**First appears in**: Lab 4.2.1

### Red Teaming
**What**: Systematically attacking your own system to find vulnerabilities before real attackers do.
**Why it matters**: Proactive security - find and fix issues in development, not production.
**First appears in**: Lab 4.2.3

### Safety Taxonomy
**What**: Categorization of harmful content types (violence, illegal activity, etc.) used by safety classifiers.
**Why it matters**: Standardized categories enable consistent safety filtering across applications.
**First appears in**: Lab 4.2.2

### Hallucination
**What**: When an LLM confidently generates false or fabricated information.
**Why it matters**: Creates liability and erodes user trust - especially critical in high-stakes domains.
**First appears in**: Lab 4.2.4

### Model Card
**What**: Standardized documentation covering a model's intended use, limitations, safety evaluation, and biases.
**Why it matters**: Required for responsible deployment - regulatory bodies increasingly require this.
**First appears in**: Lab 4.2.6

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 4.1              ──►  Module 4.2           ──►   Module 4.3
Multimodal AI                AI Safety                  MLOps
[VLMs, generation]           [Guardrails, red           [Evaluation,
                              teaming, benchmarks]       tracking]
```

**Builds on**:
- LLM inference from Domain 3 (now adding safety layers)
- Agent capabilities from Module 3.6 (agents need safety controls)
- Multimodal systems from Module 4.1 (VLMs have safety considerations too)

**Prepares for**:
- Module 4.3 MLOps includes safety metrics in experiment tracking
- Module 4.4 deployment requires safety documentation
- Module 4.6 capstone must include safety evaluation

---

## Why This Module is P0 Critical

| Reason | Impact |
|--------|--------|
| **EU AI Act** | Legal requirement in Europe - violations = massive fines |
| **NIST AI RMF** | US federal standard - required for government contracts |
| **Industry Standard** | Major tech companies all have AI safety teams |
| **User Trust** | Safety failures destroy reputation quickly |
| **Liability** | Unsafe AI outputs can lead to lawsuits |

---

## Recommended Approach

**Standard Path** (15 hours):
1. Start with overview (00) - understand the landscape
2. Lab 4.2.1 NeMo Guardrails - the main defense framework
3. Lab 4.2.2 Llama Guard - complement to NeMo
4. Lab 4.2.3 Red Teaming - attack what you built
5. Labs 4.2.4-5 - measure safety systematically
6. Lab 4.2.6 - document everything

**Quick Path** (if experienced, 8-10 hours):
1. Skim overview
2. Focus on Labs 4.2.1 + 4.2.3 (guardrails + red teaming)
3. Complete Lab 4.2.6 (model card is required)

**Deep-Dive Path** (20+ hours):
1. All labs with extended exercises
2. Implement all OWASP LLM Top 10 mitigations
3. Create comprehensive red team report
4. Publish multiple model cards

---

## Key Regulatory Context

### EU AI Act Risk Categories

| Risk Level | Examples | Requirements |
|------------|----------|--------------|
| Unacceptable | Social scoring, emotion recognition | Banned |
| High | Medical, legal, employment AI | Strict compliance |
| Limited | Chatbots, AI-generated content | Transparency |
| Minimal | Spam filters, games | No requirements |

### NIST AI RMF Functions

1. **GOVERN**: Establish AI risk management culture
2. **MAP**: Identify and assess AI risks
3. **MEASURE**: Analyze and track risks
4. **MANAGE**: Prioritize and respond to risks

---

## Before You Start

- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
- See [ELI5.md](./ELI5.md) for concept explanations
