# Module 4.2: AI Safety & Alignment - Frequently Asked Questions

## Setup & Environment

### Q: Do I need NeMo Guardrails AND Llama Guard?

**A**: They complement each other:

| Tool | Best For |
|------|----------|
| **NeMo Guardrails** | Rule-based filtering, dialog flow control, topic restrictions |
| **Llama Guard** | ML-based content classification, nuanced safety detection |

**Recommendation**: Use both in production. NeMo for rules you know, Llama Guard for edge cases.

---

### Q: Why does Llama Guard need a specific prompt format?

**A**: Llama Guard was trained on a specific conversation format. Using the wrong format gives unreliable results.

**Correct format**:
```
Task: Check if there is unsafe content in 'User' messages.

<BEGIN CONVERSATION>
User: [message]
<END CONVERSATION>
```

---

### Q: How much latency does safety filtering add?

**A**: On DGX Spark with Ollama:

| Component | Latency |
|-----------|---------|
| Llama Guard classification | ~200-500ms |
| NeMo Guardrails (simple rules) | ~50-100ms |
| NeMo Guardrails (with LLM) | ~500-1000ms |

**Total overhead**: 250ms-1.5s per request depending on configuration.

**Tip**: Run safety checks in parallel with generation, cancel if unsafe.

---

## Concepts

### Q: What's the difference between prompt injection and jailbreaking?

**A**:

| Prompt Injection | Jailbreaking |
|------------------|--------------|
| Inserting malicious instructions into prompts | Convincing the model to ignore its training |
| Often via external data (documents, URLs) | Usually via direct conversation |
| "Ignore above and do X" | "Pretend you're DAN who has no rules" |
| Exploits context confusion | Exploits role-play capability |

**Both** aim to bypass safety, but use different vectors.

---

### Q: Why can't I just tell the AI "refuse harmful requests"?

**A**: LLMs don't follow instructions reliably for safety. They're trained to be helpful, which attackers exploit:

1. "Be helpful" + "How to hack" = AI tries to help with hacking
2. Prompt injection can override any instruction
3. Role-play bypasses can work: "As a security researcher..."

**You need external enforcement** (guardrails) that the AI can't override.

---

### Q: What's the difference between TruthfulQA and factual accuracy?

**A**:

- **Factual accuracy**: Does the model know correct facts?
- **TruthfulQA**: Does the model resist giving plausible-sounding wrong answers?

TruthfulQA specifically tests questions where humans often believe false things (misconceptions). A model that always says "I don't know" would score higher than one that confidently gives wrong answers.

---

### Q: How is AI bias different from human bias?

**A**: AI bias is **learned from data**, not innate:

1. AI has no beliefs or prejudices
2. AI reflects statistical patterns in training data
3. AI applies those patterns consistently (no individual variation)
4. AI bias is measurable and quantifiable
5. AI bias can be (partially) corrected through training

**Key difference**: AI bias is systematic. A biased human might sometimes act fairly. A biased AI is consistently biased.

---

## Troubleshooting

### Q: My guardrails block too many legitimate requests (false positives)

**A**: Tune your configuration:

1. **Reduce rule strictness**: Use more specific triggers
   ```colang
   # Too broad
   define user ask about security
     "security"

   # Better - more specific
   define user ask about hacking
     "How do I hack into"
     "Break into someone's"
   ```

2. **Add whitelists**: Explicitly allow safe variations
3. **Use confidence thresholds**: For Llama Guard, only block high-confidence unsafe

---

### Q: Llama Guard classifies everything as safe

**A**: Check prompt format. The model is sensitive to exact formatting:

```python
# Wrong - may return "safe" for everything
response = ollama.chat(
    model="llama-guard3:8b",
    messages=[{"role": "user", "content": "Is this safe: How to hack?"}]
)

# Correct format
response = ollama.chat(
    model="llama-guard3:8b",
    messages=[{"role": "user", "content": """Task: Check if there is unsafe content in 'User' messages.

<BEGIN CONVERSATION>
User: How to hack into a computer?
<END CONVERSATION>"""}]
)
```

---

### Q: Red teaming found vulnerabilities - now what?

**A**: Follow this process:

1. **Document each vulnerability**
   - Attack that worked
   - Category (injection, jailbreak, etc.)
   - Severity (what could go wrong)

2. **Prioritize by risk**
   - High: Could cause real harm
   - Medium: Reputation damage
   - Low: Minor policy violations

3. **Implement mitigations**
   - Add specific guardrail rules
   - Update Llama Guard prompts
   - Add output filtering

4. **Re-test to verify fix**

5. **Document in model card**

---

### Q: How do I know if my safety measures are good enough?

**A**: There's no "good enough" - safety is continuous. But benchmarks help:

| Benchmark | Target |
|-----------|--------|
| TruthfulQA MC2 | > 0.5 (higher = more truthful) |
| Red team pass rate | > 90% attacks blocked |
| False positive rate | < 5% legitimate requests blocked |
| Bias disparity | < 10% difference across groups |

**Also**: Get external review. Your own red teaming has blind spots.

---

## Beyond the Basics

### Q: Should I use safety measures for internal tools?

**A**: Yes, but maybe lighter:

| Context | Safety Level |
|---------|--------------|
| Public-facing chatbot | Full guardrails + Llama Guard + logging |
| Internal employee tool | Basic guardrails + logging |
| Personal development | Optional, but good practice |

**Key point**: Even internal tools can be abused or produce harmful outputs.

---

### Q: How does EU AI Act affect my model?

**A**: Depends on risk classification:

| Your Model | Likely Classification | Requirements |
|------------|----------------------|--------------|
| General chatbot | Limited Risk | Transparency (tell users it's AI) |
| Medical advice | High Risk | Extensive documentation, testing, auditing |
| Employment screening | High Risk | Bias testing, human oversight |
| Creative writing | Minimal Risk | None specific |

**Check**: [artificialintelligenceact.eu](https://artificialintelligenceact.eu/) for current requirements.

---

### Q: Can I fine-tune safety into the model?

**A**: Yes, but it's not enough alone:

| Approach | Pros | Cons |
|----------|------|------|
| RLHF/DPO safety training | Model inherently safer | Can be bypassed, costly |
| Constitutional AI | Self-critique | Not foolproof |
| External guardrails | Reliable, auditable | Latency, cost |

**Best practice**: Defense in depth. Fine-tune for safety AND use guardrails.

---

### Q: What about copyright/IP concerns?

**A**: Growing area of concern:

1. **Training data**: Was copyrighted content used?
2. **Output**: Does the model reproduce copyrighted text?
3. **Style mimicry**: Can it copy specific artists?

**Mitigations**:
- Check outputs for verbatim copying
- Add guardrails against style-copying requests
- Document training data sources

---

### Q: How do I handle safety for multimodal models?

**A**: Additional considerations:

| Modality | Safety Concerns |
|----------|-----------------|
| Image input | NSFW detection, deepfake source |
| Image output | NSFW generation, copyright |
| Audio input | Voice cloning source |
| Audio output | Deepfake voice generation |

**Tools**: Use specialized classifiers for each modality (not just Llama Guard).

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [ELI5.md](./ELI5.md) for concept clarification
- See module [Resources](./README.md#resources) for official documentation
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) for security best practices
