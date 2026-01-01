# Module 4.6: Capstone Project - Quickstart

## Time: ~15 minutes

## What You'll Do

Quickly validate your project idea and create a minimal working prototype.

## Before You Start

- [ ] Completed Domains 1-4 modules
- [ ] Chosen a project option (A, B, C, or D)
- [ ] DGX Spark access

## Let's Go!

### Step 1: Define Your MVP

Pick the simplest possible version that demonstrates your concept:

**Option A (Domain AI Assistant)**:
- Single-turn Q&A (not full conversation)
- One small document in RAG
- Basic safety check

**Option B (Multimodal Document)**:
- One PDF processing
- Simple extraction
- Text output only

**Option C (Agent Swarm)**:
- Two agents only
- One tool each
- Simple handoff

**Option D (Training Pipeline)**:
- One fine-tuning run
- Single evaluation metric
- Manual tracking

### Step 2: Create Project Structure

```bash
mkdir capstone && cd capstone

# Create structure
mkdir -p src data models notebooks docs

# Initialize files
touch src/__init__.py
touch src/main.py
touch requirements.txt
touch README.md
```

### Step 3: Write Minimal Main Script

**Example for Option A (Domain Assistant)**:

```python
# src/main.py
import ollama
from nemo_guardrails import LLMRails, RailsConfig

# 1. Simple RAG context (hardcoded for MVP)
CONTEXT = """
Company Policy: Employees get 20 vacation days per year.
Vacation requests must be submitted 2 weeks in advance.
"""

# 2. Create simple guardrails config
config = RailsConfig.from_path("config/")

# 3. Chat function
def chat(question: str) -> str:
    prompt = f"""Answer based on this context:
{CONTEXT}

Question: {question}
Answer:"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# 4. Test
if __name__ == "__main__":
    answer = chat("How many vacation days do I get?")
    print(answer)
```

### Step 4: Run Your MVP

```bash
python src/main.py
```

**Expected output**:
```
Employees get 20 vacation days per year.
```

### Step 5: Validate the Concept

Ask yourself:
- Does the core idea work?
- What's the hardest part to build?
- What's the most impressive feature to demo?

## You Did It!

You have a working prototype! Now expand it:

1. **Week 35**: Complete planning and architecture
2. **Weeks 36-37**: Build core components
3. **Week 38**: Integration
4. **Week 39**: Optimization and safety evaluation
5. **Week 40**: Documentation and presentation

## Next Steps

1. **Fill out proposal**: Use `templates/project-proposal.md`
2. **Plan architecture**: Draw system diagram
3. **Start documentation**: Begin technical report
4. **Full guide**: See [STUDY_GUIDE.md](./STUDY_GUIDE.md)
