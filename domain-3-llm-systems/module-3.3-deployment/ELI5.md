# Module 3.3: Model Deployment & Inference Engines - ELI5 Explanations

> **What is ELI5?** These explanations use everyday analogies to build intuition.
> Read these BEFORE diving into the technical material—they'll make everything click faster.

---

## 🧒 Inference Engines: The Restaurant Kitchen

### The Jargon-Free Version
An inference engine is software that runs your LLM efficiently. Different engines are optimized for different things, like restaurants specializing in different cuisines.

### The Analogy
**Inference engines are like different types of restaurant kitchens...**

You have a recipe (your model). How you prepare it depends on your kitchen:

- **Ollama**: Home kitchen. Easy to set up, great for personal use. Not the fastest, but convenient.

- **vLLM**: Fast-food kitchen. Continuous batching means orders are grouped and processed together. High throughput for many customers.

- **TensorRT-LLM**: Michelin-star kitchen. Every step is optimized by the chef (NVIDIA). Fastest prep (prefill), but takes time to set up.

- **SGLang**: Modern kitchen with smart caching. If you ordered the same appetizer yesterday, it's already prepped (RadixAttention).

- **llama.cpp**: Food truck. Lightweight, runs anywhere, even without fancy equipment (GPU optional).

### Quick Decision Guide
```
Just learning?         ──► Ollama
Need high throughput?  ──► vLLM
Chat with shared prompts? ──► SGLang
Maximum speed?         ──► TensorRT-LLM
Run on CPU?            ──► llama.cpp
```

### When You're Ready for Details
→ See: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for engine comparison table

---

## 🧒 Continuous Batching: Serving Everyone at Once

### The Jargon-Free Version
Instead of waiting for one request to finish before starting the next, continuous batching processes multiple requests simultaneously, filling GPU capacity.

### The Analogy
**Continuous batching is like a smart conveyor belt sushi restaurant...**

**Without batching (one at a time)**:
- Customer 1 orders, chef makes it, customer 1 eats
- Customer 2 waits... and waits...
- Very slow, chef is often idle

**With continuous batching**:
- Customers sit at a conveyor belt
- Chef puts out sushi continuously
- When customer 1's plate leaves, a new customer joins
- Chef is always busy, everyone gets served faster

In LLMs, the "conveyor belt" is GPU memory. As one request finishes generating tokens, a new request can join the batch. The GPU is always working.

### Key Insight
Continuous batching is why vLLM can handle 100 users simultaneously when naive serving can only handle 1-2.

### When You're Ready for Details
→ See: [Lab 3.3.3](./labs/lab-3.3.3-vllm-continuous-batching.ipynb) for throughput benchmarks

---

## 🧒 PagedAttention: Smart Memory Management

### The Jargon-Free Version
PagedAttention stores attention memory (KV cache) in flexible "pages" instead of one big block, preventing memory waste.

### The Analogy
**PagedAttention is like a smart notebook vs. a fixed notebook...**

**Without PagedAttention (fixed allocation)**:
- You get a 100-page notebook for each conversation
- Short conversation? 90 pages wasted
- Many conversations? Not enough notebooks for everyone

**With PagedAttention (flexible allocation)**:
- You get loose-leaf paper, one page at a time
- Short conversation uses 10 pages
- Long conversation uses 100 pages
- Pages from finished conversations are recycled immediately

This is why vLLM can handle more concurrent users—it doesn't waste memory on unused space.

### When You're Ready for Details
→ See: [Lab 3.3.3](./labs/lab-3.3.3-vllm-continuous-batching.ipynb) for memory analysis

---

## 🧒 Speculative Decoding: The Guess-and-Check Speedup

### The Jargon-Free Version
Use a small, fast model to guess the next several words, then verify all guesses at once with the big model. If guesses are right, you skip ahead!

### The Analogy
**Speculative decoding is like a typing assistant...**

Imagine you're typing an email and your assistant starts suggesting the next words:

**Without speculation**:
- You type "T" → wait → "Thank" → wait → "Thank you" → wait → "Thank you for"...
- Each word takes one "think cycle"

**With speculation (draft model guesses)**:
- Assistant suggests: "Thank you for your email regarding the meeting"
- You verify: "Yes, that's exactly what I wanted!"
- 8 words in 1 "think cycle"!

If the guess is wrong? You accept the correct prefix and try again. Wrong guesses cost nothing extra.

### A Visual
```
Without speculation:    T → Th → Tha → Than → Thank → Thank y → ...
                        ↓    ↓     ↓     ↓      ↓        ↓
                       [think][think][think][think][think][think]

With speculation:       "Thank you for your email" (draft guess)
                                    ↓
                        [verify all at once] → Accept 5 words!
```

### The Numbers
- **Medusa**: 2-3x speedup (adds prediction heads, no separate model)
- **EAGLE**: Similar speedup, better for long sequences
- **Best for**: Interactive chat, predictable text patterns

### When You're Ready for Details
→ See: [Lab 3.3.4](./labs/lab-3.3.4-speculative-decoding.ipynb) for implementation

---

## 🧒 RadixAttention: Remembering Shared Work

### The Jargon-Free Version
SGLang's RadixAttention caches the computation for shared prefixes (like system prompts), so repeated requests don't redo the same work.

### The Analogy
**RadixAttention is like a photocopier for legal documents...**

Imagine 100 people need a contract. Each contract has:
- 50 pages of standard legal text (same for everyone)
- 1 page of custom terms (different for each person)

**Without RadixAttention**:
- Print all 51 pages × 100 people = 5,100 pages printed

**With RadixAttention**:
- Print 50 standard pages once, photocopy them
- Print 1 custom page per person
- Total: 50 + 100 = 150 pages of actual work

For LLMs, the "system prompt" is like the standard legal text. All users share it, so why process it 100 times? RadixAttention caches the result.

### The Numbers
- **Chat apps with system prompts**: 29-45% faster than vLLM
- **Batch processing with shared context**: Even bigger gains

### When You're Ready for Details
→ See: [Lab 3.3.2](./labs/lab-3.3.2-sglang-deployment.ipynb) for SGLang setup

---

## 🧒 Prefill vs Decode: Two Phases of Generation

### The Jargon-Free Version
LLM generation has two phases: reading the input (prefill) and writing the output (decode). They're bottlenecked differently.

### The Analogy
**It's like reading a book vs. writing a response...**

**Prefill (reading the prompt)**:
- Process all input tokens at once
- Highly parallel—like reading a page in one glance
- GPU compute-bound: Faster GPU = faster prefill
- Speed measured in "prefill tok/s"

**Decode (generating output)**:
- Generate one token at a time
- Sequential—like writing one word after another
- Memory bandwidth-bound: Faster memory = faster decode
- Speed measured in "decode tok/s"

Different engines excel at different phases:
- **TensorRT-LLM**: Best prefill (optimized for compute)
- **llama.cpp**: Best decode (optimized for memory access)
- **vLLM/SGLang**: Good balance of both

### Key Insight
A 10,000 tok/s prefill engine might only decode at 40 tok/s. That's normal! They're different bottlenecks.

### When You're Ready for Details
→ See: [Lab 3.3.1](./labs/lab-3.3.1-engine-benchmark.ipynb) for performance comparison

---

## 🧒 Medusa Heads: Multiple Predictions at Once

### The Jargon-Free Version
Medusa adds extra "prediction heads" to the model that guess multiple future tokens simultaneously. No separate draft model needed.

### The Analogy
**Medusa heads are like a hydra making multiple guesses...**

A normal model has one head that predicts the next token. Medusa adds 3-5 extra heads that predict:
- Head 1: Next token (same as normal)
- Head 2: Token after that
- Head 3: Two tokens ahead
- ...and so on

All heads predict at the same time (one forward pass). Then we verify which predictions form a valid sequence.

```
Normal:  [Model] → "The"
                        ↓
         [Model] → "cat"
                        ↓
         [Model] → "sat"

Medusa:  [Model] → "The" + "cat" + "sat" + "on"
         (all predictions at once!)
         Verify: Accept "The cat sat" → Skip 3 forward passes!
```

### Why This Works
Later tokens are easier to predict because they have more context. "sat" is predictable after "The cat" even if you're guessing ahead.

### When You're Ready for Details
→ See: [Lab 3.3.4](./labs/lab-3.3.4-medusa-speculative-decoding.ipynb) for Medusa implementation

---

## 🔗 From ELI5 to Technical

| ELI5 Term | Technical Term | Where to Learn More |
|-----------|----------------|---------------------|
| "Restaurant kitchen" | Inference engine | Lab 3.3.1 |
| "Conveyor belt" | Continuous batching | Lab 3.3.3 |
| "Loose-leaf paper" | PagedAttention | Lab 3.3.3 |
| "Typing assistant" | Speculative decoding | Lab 3.3.4 |
| "Photocopier" | RadixAttention prefix caching | Lab 3.3.2 |
| "Reading vs writing" | Prefill vs decode | Lab 3.3.1 |
| "Multiple guesses" | Medusa heads | Lab 3.3.4 |

---

## 💡 The "Explain It Back" Test

You truly understand these concepts when you can explain them to someone else without jargon. Try explaining:

1. Why vLLM can handle 100 users but naive serving can't
2. How speculative decoding gets 2-3x speedup without better hardware
3. Why system prompts are faster with SGLang
