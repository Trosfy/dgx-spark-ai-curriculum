# Module 4.6: Capstone Project - Frequently Asked Questions

## Project Selection

### Q: Which project option should I choose?

**A**: Choose based on your interests and career goals:

| If you want to... | Choose |
|-------------------|--------|
| Work on LLM applications | Option A: Domain Assistant |
| Work with documents/images | Option B: Multimodal |
| Build complex systems | Option C: Agent Swarm |
| Focus on ML infrastructure | Option D: Training Pipeline |

**Recommendation**: Pick what excites you - you'll spend 40+ hours on it.

---

### Q: Can I combine project options?

**A**: Yes, but be careful about scope. Good combinations:

| Combination | Works Well | Why |
|-------------|------------|-----|
| A + B | Yes | Document RAG with fine-tuning |
| A + C | Yes | Agents with domain expertise |
| C + Safety | Yes | Required for Option C |
| All options | No | Too broad, won't finish |

Keep core focus on one option with elements from others.

---

### Q: What if I want to do something not listed?

**A**: Custom projects are allowed if they:

1. Demonstrate equivalent complexity
2. Use DGX Spark capabilities
3. Include required deliverables
4. Have clear evaluation criteria

Get approval before starting a custom project.

---

## Scope and Planning

### Q: How do I know if my scope is too big?

**A**: Signs your project is too ambitious:

- More than 5 major components
- Requires data you don't have
- Needs external APIs you can't access
- Would take a team months to build
- You can't explain it in 2 sentences

**Rule of thumb**: If you can't build an MVP in week 1, scope down.

---

### Q: How do I know if my scope is too small?

**A**: Signs your project is too simple:

- Single API call wrapper
- No model training or fine-tuning
- No evaluation beyond "it works"
- Could be done in a weekend
- Doesn't use DGX Spark's unique capabilities

**Rule of thumb**: Should take 40+ hours to complete properly.

---

### Q: What's a realistic timeline?

**A**: Based on past projects:

| Phase | Expected | Buffer for Issues |
|-------|----------|-------------------|
| Planning | 1 week | + 2 days |
| Foundation | 2 weeks | + 3-4 days |
| Integration | 1 week | + 2 days |
| Optimization | 1 week | + 2 days |
| Documentation | 1 week | + 1 day |

Build in buffer time for unexpected issues.

---

## Technical Questions

### Q: How do I leverage DGX Spark's 128GB memory?

**A**: Great ways to use the unified memory:

1. **Larger models**: Run 70B+ models without quantization
2. **Bigger batch sizes**: Faster fine-tuning
3. **Multiple models**: Keep several loaded simultaneously
4. **Large documents**: Process entire document sets in memory
5. **Agent parallelism**: Run multiple agents concurrently

Document your memory usage in the technical report.

---

### Q: What evaluation metrics should I use?

**A**: Depends on project type:

| Project | Metrics |
|---------|---------|
| Option A | Task accuracy, latency, safety pass rate |
| Option B | Extraction F1, comprehension scores |
| Option C | Task success rate, safety violations, latency |
| Option D | Model improvement, pipeline throughput |

Always include:
- Performance metrics (accuracy, F1, etc.)
- Latency measurements
- Safety evaluation (if applicable)
- Cost analysis (if cloud-deployed)

---

### Q: How extensive should safety evaluation be?

**A**: Minimum requirements by project:

| Project | Safety Requirements |
|---------|---------------------|
| Option A | Guardrails config + test suite + model card |
| Option B | Input validation + output filtering |
| Option C | Agent action limits + human-in-loop + red teaming |
| Option D | Data quality checks + model validation |

Safety is 15% of the grade - don't skip it.

---

### Q: What if I run out of compute resources?

**A**: Strategies to reduce compute:

1. **Quantize models**: Use 4-bit instead of 16-bit
2. **Smaller models**: 8B instead of 70B for development
3. **Shorter training**: Fewer epochs, validate early
4. **Batch efficiently**: Optimize batch sizes
5. **Use checkpoints**: Resume training, don't restart

---

## Deliverables

### Q: How long should the technical report be?

**A**: 15-20 pages, focused content:

| Section | Pages |
|---------|-------|
| Introduction/Problem | 1-2 |
| Related Work | 1-2 |
| System Architecture | 3-4 |
| Implementation | 4-5 |
| Evaluation | 3-4 |
| Discussion/Lessons | 2-3 |
| Appendix (optional) | 2-3 |

Quality over quantity - concise is better than padded.

---

### Q: What should the demo showcase?

**A**: Focus on:

1. **Core functionality**: Main feature working smoothly
2. **Best examples**: Pre-selected inputs that work well
3. **Edge case handling**: Show graceful failure
4. **Speed**: Optimize for demo performance

Don't demo:
- Features that might break
- Slow operations without progress indicators
- Raw error messages

---

### Q: What should be in the model card?

**A**: Essential sections:

```markdown
# Model Card: [Your Model Name]

## Model Details
- Base model
- Fine-tuning method
- Training data summary

## Intended Use
- Primary use case
- Out-of-scope uses

## Training Data
- Data sources
- Data size
- Data processing

## Evaluation
- Metrics and results
- Safety evaluation results
- Limitations

## Ethical Considerations
- Potential biases
- Mitigation strategies
- Recommendations for use
```

---

### Q: How should I organize my code repository?

**A**: Recommended structure:

```
capstone/
├── README.md           # Setup instructions
├── requirements.txt    # Dependencies
├── Dockerfile          # Container build
├── src/
│   ├── __init__.py
│   ├── main.py         # Entry point
│   ├── model.py        # Model code
│   ├── data.py         # Data processing
│   └── utils.py        # Utilities
├── tests/
│   ├── test_model.py
│   └── test_data.py
├── notebooks/          # Development notebooks
├── configs/            # Configuration files
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_CARD.md
└── demo/
    └── app.py          # Gradio/Streamlit demo
```

---

## Common Issues

### Q: My project isn't working as expected

**A**: Debugging strategy:

1. **Isolate the problem**: Which component fails?
2. **Check basics**: Data format, model loading, API calls
3. **Review logs**: Error messages, stack traces
4. **Simplify**: Does minimal example work?
5. **Reference modules**: Did it work in earlier labs?

Document issues and solutions in your report.

---

### Q: I'm running behind schedule

**A**: Recovery strategies:

1. **Reduce scope**: Cut nice-to-have features
2. **Simplify evaluation**: Fewer metrics, smaller test set
3. **Streamline docs**: Focus on key sections
4. **Use existing code**: Adapt from earlier modules
5. **Ask for help**: Don't struggle alone

What to cut first:
- Extra features
- Extensive ablations
- Perfect documentation

What to keep:
- Core functionality
- Safety evaluation
- Demo that works

---

## Still Have Questions?

- Review [STUDY_GUIDE.md](./STUDY_GUIDE.md) for detailed timeline
- Check templates in `templates/` folder
- Review earlier modules for technique refreshers
