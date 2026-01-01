# Module 4.5: Demo Building - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Build complex Gradio interfaces** with Blocks API, tabs, and custom layouts
2. **Create multi-page Streamlit apps** with proper state management
3. **Integrate RAG, agents, and chat** in polished demos
4. **Deploy to free platforms** (Hugging Face Spaces, Streamlit Cloud)

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-4.5.1-complete-rag-demo.ipynb | Gradio Blocks | ~3h | Full RAG demo on HF Spaces |
| 2 | lab-4.5.2-agent-playground.ipynb | Streamlit Advanced | ~3h | Agent playground with tool viz |
| 3 | lab-4.5.3-portfolio-demo.ipynb | Portfolio Polish | ~2h | Capstone demo ready |

**Total Time**: ~8 hours

---

## Core Concepts

### Gradio Blocks API
**What**: Low-level API for building complex, custom Gradio layouts.
**Why it matters**: Simple `gr.Interface` can't handle tabs, columns, or dynamic updates.
**First appears in**: Lab 4.5.1

### Session State
**What**: Persistent data across user interactions and page reloads.
**Why it matters**: Demos need to remember conversation history, user preferences.
**First appears in**: Labs 4.5.1-2

### Progressive Disclosure
**What**: Showing simple interface first, revealing complexity on demand.
**Why it matters**: Good demos don't overwhelm users with options.
**First appears in**: Lab 4.5.1

### Demo vs Production
**What**: Understanding when to cut corners and when to polish.
**Why it matters**: Demos prove concepts; production serves users. Different standards.
**First appears in**: Lab 4.5.3

---

## How This Module Connects

```
Previous                    This Module                 Next
---------------------------------------------------------------------
Module 4.4              -->  Module 4.5           -->   Module 4.6
Containerization             Demo Building              Capstone
[Docker, K8s,                [Gradio, Streamlit,        [End-to-end
 cloud platforms]             HF Spaces]                 portfolio]
```

**Builds on**:
- Containerization from Module 4.4 (demos can be containerized)
- RAG systems from Module 3.5 (showcase in demos)
- Agents from Module 3.6 (visualize in playgrounds)

**Prepares for**:
- Module 4.6 capstone requires polished demo
- Portfolio projects need public demos

---

## Gradio vs Streamlit Decision

| Use Gradio When | Use Streamlit When |
|-----------------|-------------------|
| ML model demo | Data dashboards |
| Quick prototype (minutes) | Multi-page apps |
| Chat interface | Complex state management |
| Hugging Face deployment | Charts and visualizations |
| Simple file inputs/outputs | Form-heavy applications |

---

## Recommended Approach

**Standard Path** (8 hours):
1. Lab 4.5.1: Master Gradio Blocks API
2. Lab 4.5.2: Build Streamlit multi-page app
3. Lab 4.5.3: Polish capstone demo

**Quick Path** (if experienced, 4 hours):
1. Skim Lab 4.5.1, focus on Blocks patterns
2. Lab 4.5.2: Agent playground
3. Lab 4.5.3: Portfolio demo

**Portfolio-focused Path**:
1. Lab 4.5.1: Full RAG demo
2. Skip Lab 4.5.2
3. Lab 4.5.3: Extra time on polish

---

## Before You Start

- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
- See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for Gradio/Streamlit patterns
- See [FAQ.md](./FAQ.md) for common questions
