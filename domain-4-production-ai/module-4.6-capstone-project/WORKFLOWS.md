# Module 4.6: Capstone Project - Workflow Cheatsheets

## Workflow 1: Project Planning (Week 1)

### When to Use
Use this workflow at the start of your capstone to define scope and get approval.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Choose Project Option                               │
├─────────────────────────────────────────────────────────────┤
│ □ Review all four options (A-D) in README.md                │
│ □ Consider your interests and prior module strengths        │
│ □ Check resource requirements match your setup              │
│                                                             │
│ Decision matrix:                                            │
│ - Option A: Like LLMs + RAG? Strong in Module 3.1/3.5?     │
│ - Option B: Like vision? Strong in Module 4.1?             │
│ - Option C: Like agents? Strong in Module 3.6?             │
│ - Option D: Like training? Strong in Module 3.1?           │
│                                                             │
│ ✓ Checkpoint: Option selected and justified                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Define Use Case                                     │
├─────────────────────────────────────────────────────────────┤
│ □ Identify specific domain/problem                          │
│ □ Define target users                                       │
│ □ List 3-5 key features                                     │
│ □ Write one-sentence project description                    │
│                                                             │
│ Template:                                                   │
│ "A [type of system] for [users] that [solves problem]       │
│  using [key technologies]."                                 │
│                                                             │
│ ✓ Checkpoint: Can explain project in 30 seconds             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Draft Architecture                                  │
├─────────────────────────────────────────────────────────────┤
│ □ Identify major components                                 │
│ □ Draw data flow diagram                                    │
│ □ List external dependencies                                │
│ □ Identify potential risks                                  │
│                                                             │
│ Components to consider:                                     │
│ - Model (which, what size, quantization?)                  │
│ - Data (source, format, storage?)                          │
│ - Retrieval (ChromaDB, Pinecone?)                          │
│ - Safety (guardrails approach?)                            │
│ - Demo (Gradio or Streamlit?)                              │
│                                                             │
│ ✓ Checkpoint: Architecture diagram completed                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Create Timeline                                     │
├─────────────────────────────────────────────────────────────┤
│ □ Break into weekly milestones                              │
│ □ Identify dependencies between tasks                       │
│ □ Add buffer time (20% extra)                               │
│ □ Mark "must-have" vs "nice-to-have"                       │
│                                                             │
│ Standard timeline:                                          │
│ Week 1: Planning + Data collection                          │
│ Week 2: Core implementation                                 │
│ Week 3: Integration + Testing                               │
│ Week 4: Optimization + Safety                               │
│ Week 5: Documentation + Demo                                │
│                                                             │
│ ✓ Checkpoint: Timeline with specific deliverables           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Write Proposal                                      │
├─────────────────────────────────────────────────────────────┤
│ □ Use templates/project-proposal.md                         │
│ □ Fill all sections                                         │
│ □ Review for completeness                                   │
│ □ Submit for approval                                       │
│                                                             │
│ ✓ Checkpoint: Proposal submitted and approved               │
└─────────────────────────────────────────────────────────────┘
```

### Common Pitfalls

| At Step | Watch Out For |
|---------|---------------|
| 1 | Choosing based on what sounds cool, not your strengths |
| 2 | Scope too vague ("make an AI assistant") |
| 3 | Missing safety/evaluation components |
| 4 | Not allocating time for documentation |

---

## Workflow 2: Data Preparation

### When to Use
Use after proposal approval to prepare your training/RAG data.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Collect Raw Data                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Identify all data sources                                 │
│ □ Download/scrape/export data                               │
│ □ Store in data/raw/                                        │
│ □ Document source and license                               │
│                                                             │
│ Common sources:                                             │
│ - PDFs, documentation                                       │
│ - Web pages (with permission)                               │
│ - Existing datasets (HuggingFace)                          │
│ - Synthetic data (LLM-generated)                           │
│                                                             │
│ ✓ Checkpoint: All raw data downloaded                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Clean and Preprocess                                │
├─────────────────────────────────────────────────────────────┤
│ □ Remove duplicates                                         │
│ □ Fix encoding issues                                       │
│ □ Handle missing values                                     │
│ □ Normalize formats                                         │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ def clean_text(text):                                       │
│     text = text.strip()                                     │
│     text = " ".join(text.split())  # Normalize whitespace  │
│     return text                                             │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Clean data in data/processed/                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Format for Task                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ For Fine-tuning (SFT):                                      │
│ ```json                                                     │
│ {"messages": [                                              │
│   {"role": "user", "content": "..."},                       │
│   {"role": "assistant", "content": "..."}                   │
│ ]}                                                          │
│ ```                                                         │
│                                                             │
│ For RAG:                                                    │
│ ```python                                                   │
│ chunks = text_splitter.split_text(document)                 │
│ vectordb.add_texts(chunks, metadatas=[...])                 │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Data formatted correctly for use              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Split and Validate                                  │
├─────────────────────────────────────────────────────────────┤
│ □ Create train/validation/test splits                       │
│ □ Verify no data leakage                                    │
│ □ Check class balance (if applicable)                       │
│ □ Sample and manually inspect                               │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ from datasets import Dataset                                │
│ dataset = Dataset.from_list(data)                           │
│ splits = dataset.train_test_split(test_size=0.1)            │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Splits saved, quality verified                │
└─────────────────────────────────────────────────────────────┘
```

### Success Criteria

- [ ] All data collected and documented
- [ ] Data cleaned and preprocessed
- [ ] Format matches task requirements
- [ ] Train/val/test splits created
- [ ] Manual inspection passed

---

## Workflow 3: Model Development

### When to Use
Use during core implementation phase (Weeks 2-3).

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Establish Baseline                                  │
├─────────────────────────────────────────────────────────────┤
│ □ Load base model                                           │
│ □ Test on sample inputs                                     │
│ □ Run baseline benchmarks                                   │
│ □ Document baseline performance                             │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ # Baseline evaluation                                       │
│ baseline_results = evaluate_model(base_model, test_set)     │
│ mlflow.log_metrics({"baseline_accuracy": baseline_results}) │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Baseline metrics logged                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Implement Core Feature                              │
├─────────────────────────────────────────────────────────────┤
│ □ Start with simplest implementation                        │
│ □ Get end-to-end working                                    │
│ □ Test with single example                                  │
│ □ Fix obvious issues                                        │
│                                                             │
│ "Make it work, make it right, make it fast"                │
│ (Focus on "make it work" first)                            │
│                                                             │
│ ✓ Checkpoint: Core feature working end-to-end               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Train/Fine-tune (if applicable)                     │
├─────────────────────────────────────────────────────────────┤
│ □ Configure training parameters                             │
│ □ Start with small run (10 steps)                           │
│ □ Verify loss decreases                                     │
│ □ Run full training                                         │
│ □ Save checkpoints                                          │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ trainer.train()                                             │
│ trainer.save_model("models/final")                          │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Trained model saved                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Iterate and Improve                                 │
├─────────────────────────────────────────────────────────────┤
│ □ Compare to baseline                                       │
│ □ Identify weak points                                      │
│ □ Try improvements                                          │
│ □ Log all experiments                                       │
│                                                             │
│ Track with MLflow:                                          │
│ ```python                                                   │
│ with mlflow.start_run(run_name="exp-001"):                  │
│     mlflow.log_params(config)                               │
│     # ... train ...                                         │
│     mlflow.log_metrics(results)                             │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Improvement over baseline documented          │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow 4: Evaluation & Safety

### When to Use
Use during Week 4 to ensure your system is ready for deployment.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Run Standard Benchmarks                             │
├─────────────────────────────────────────────────────────────┤
│ □ Choose relevant benchmarks                                │
│ □ Run lm-evaluation-harness                                 │
│ □ Compare to baseline                                       │
│ □ Document results                                          │
│                                                             │
│ ```bash                                                     │
│ lm_eval --model hf \                                        │
│     --model_args pretrained=./models/final \                │
│     --tasks mmlu,hellaswag,truthfulqa_mc2 \                 │
│     --output_path ./results/benchmarks                      │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Benchmark results saved                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Run Safety Evaluation                               │
├─────────────────────────────────────────────────────────────┤
│ □ Test with attack prompts                                  │
│ □ Run TruthfulQA                                            │
│ □ Check for harmful outputs                                 │
│ □ Test guardrails effectiveness                             │
│                                                             │
│ ```python                                                   │
│ safety_results = evaluate_safety(model, ATTACK_PROMPTS)     │
│ print(f"Refusal rate: {safety_results['refusal_rate']}")    │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Safety evaluation documented                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Custom Evaluation                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Define domain-specific metrics                            │
│ □ Create test cases                                         │
│ □ Run evaluation                                            │
│ □ Compare to requirements                                   │
│                                                             │
│ Example metrics:                                            │
│ - RAG: retrieval precision, answer accuracy                │
│ - Agents: task completion rate, tool usage                 │
│ - Training: perplexity improvement, format compliance      │
│                                                             │
│ ✓ Checkpoint: Custom metrics passing requirements           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Create Model Card                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Use templates/model-card.md                               │
│ □ Document all evaluation results                           │
│ □ List limitations and biases                               │
│ □ Define intended use and misuse cases                      │
│                                                             │
│ ✓ Checkpoint: Model card completed                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow 5: Documentation & Demo

### When to Use
Use during final week to complete all deliverables.

### Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Build Demo Application                              │
├─────────────────────────────────────────────────────────────┤
│ □ Choose platform (Gradio or Streamlit)                     │
│ □ Create clean interface                                    │
│ □ Add error handling                                        │
│ □ Test thoroughly                                           │
│ □ Deploy to Spaces/Streamlit Cloud                          │
│                                                             │
│ ✓ Checkpoint: Demo accessible via public URL                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Write Technical Report                              │
├─────────────────────────────────────────────────────────────┤
│ □ Use templates/technical-report.md                         │
│ □ Include all sections (15-20 pages)                        │
│ □ Add figures and tables                                    │
│ □ Cite sources                                              │
│ □ Proofread                                                 │
│                                                             │
│ Sections:                                                   │
│ 1. Introduction                                             │
│ 2. Related Work                                             │
│ 3. System Architecture                                      │
│ 4. Implementation                                           │
│ 5. Evaluation                                               │
│ 6. Discussion                                               │
│ 7. Conclusion                                               │
│                                                             │
│ ✓ Checkpoint: Report complete and proofread                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Record Demo Video                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Plan 5-10 minute walkthrough                              │
│ □ Show key features                                         │
│ □ Demonstrate capabilities and limitations                  │
│ □ Record with clear audio                                   │
│                                                             │
│ Video outline:                                              │
│ 0:00 - Introduction (30s)                                   │
│ 0:30 - Problem & Solution (1min)                            │
│ 1:30 - Demo walkthrough (4min)                              │
│ 5:30 - Technical highlights (2min)                          │
│ 7:30 - Results & Conclusion (1min)                          │
│                                                             │
│ ✓ Checkpoint: Video recorded and uploaded                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Prepare Presentation                                │
├─────────────────────────────────────────────────────────────┤
│ □ Use templates/presentation-outline.md                     │
│ □ Create 15-20 slides                                       │
│ □ Include architecture diagram                              │
│ □ Add demo screenshots                                      │
│ □ Practice presentation                                     │
│                                                             │
│ ✓ Checkpoint: Presentation ready                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Finalize Repository                                 │
├─────────────────────────────────────────────────────────────┤
│ □ Clean up code                                             │
│ □ Update README                                             │
│ □ Add setup instructions                                    │
│ □ Include requirements.txt                                  │
│ □ Add Dockerfile                                            │
│ □ Final git commit                                          │
│                                                             │
│ ✓ Checkpoint: Repository ready for submission               │
└─────────────────────────────────────────────────────────────┘
```

---

## Decision Flowchart: Which Workflow?

```
                    Start
                      │
                      ▼
              ┌───────────────┐
              │ Week number?  │
              └───────┬───────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
   Week 1         Week 2-3       Week 4-5
       │              │              │
       ▼              ▼              ▼
  Workflow 1     Workflow 2&3   Workflow 4&5
  (Planning)    (Data & Model)  (Eval & Docs)
```

---

## Milestone Verification

| Week | Must Complete | How to Verify |
|------|---------------|---------------|
| 1 | Proposal approved | Written approval received |
| 2 | Data ready + MVP | Can show working demo (basic) |
| 3 | Core features work | End-to-end pipeline runs |
| 4 | Evaluation complete | Benchmark results documented |
| 5 | All deliverables | Checklist all checked |

---

## Emergency Timeline (Behind Schedule)

If you're 1+ weeks behind:

```
Week 4 (condensed):
├── Day 1-2: Finish core feature (cut scope if needed)
├── Day 3-4: Quick evaluation (limited benchmarks)
├── Day 5-6: Basic demo + documentation outline
└── Day 7: Report draft

Week 5 (documentation sprint):
├── Day 1-2: Complete report
├── Day 3: Record demo video
├── Day 4: Finalize presentation
└── Day 5-7: Polish and submit
```

**Key cuts when behind:**
- Reduce number of benchmarks
- Skip "nice-to-have" features
- Use simpler demo
- Shorten report (meet minimum 15 pages)
