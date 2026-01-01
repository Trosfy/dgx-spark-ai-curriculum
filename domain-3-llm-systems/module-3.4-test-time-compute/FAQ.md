# Module 3.4: Test-Time Compute & Reasoning - FAQ

## Concepts

### Q: When should I use Chain-of-Thought vs a reasoning model?
**A**:
- **CoT prompting**: Works with any model, no additional compute for model itself
- **Reasoning model (R1)**: Purpose-built for reasoning, higher quality on hard problems

Use CoT first. If results aren't good enough, try a reasoning model.

---

### Q: How many samples for self-consistency?
**A**:
- **5 samples**: Good balance (default)
- **3 samples**: Fast, less reliable
- **10+ samples**: High confidence, more compute

Odd numbers work best for majority voting.

---

### Q: Why do reasoning models use `<think>` tokens?
**A**: The `<think>` tokens are explicit "thinking out loud" markers. The model was trained to:
1. Generate reasoning in `<think>` blocks
2. Then provide the final answer

This makes the reasoning visible and debuggable.

---

## Troubleshooting

### Q: R1 model is too slow. What can I do?
**A**:
1. Use smaller variant (7B instead of 70B)
2. Use higher quantization (Q4_K_M)
3. Limit max_tokens to cap thinking length
4. For simple questions, use regular model

---

### Q: Self-consistency gives wrong consensus. Why?
**A**: If all paths lead to wrong answer, self-consistency won't help. Try:
1. Better prompting (more examples)
2. Stronger base model
3. CoT + self-consistency together

---

## Beyond Basics

### Q: Can I combine CoT with reasoning models?
**A**: Yes! R1 already does CoT-like reasoning, but you can add few-shot examples to guide the format. Generally, R1's built-in reasoning is sufficient.

---

### Q: How do I evaluate reasoning quality?
**A**:
1. **Accuracy**: Does it get the right answer?
2. **Reasoning validity**: Are the steps correct?
3. **Efficiency**: How many tokens to reach answer?

See Lab 3.4.4 for evaluation framework.
