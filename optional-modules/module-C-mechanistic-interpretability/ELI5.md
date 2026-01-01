# Module C: Mechanistic Interpretability - ELI5 Explanations

> **What is ELI5?** These explanations use everyday analogies to build intuition.
> Read these BEFORE diving into the technical material - they'll make everything click faster.

---

## 🧒 Mechanistic Interpretability: Looking Inside the Machine

### The Jargon-Free Version

Mechanistic interpretability means opening up a neural network to see exactly HOW it computes its answers - not just WHAT answers it gives. It's reverse-engineering the brain of an AI.

### The Analogy

**Mechanistic interpretability is like understanding a calculator vs. just using one...**

Imagine you find a mysterious calculator that always gives correct answers:
- **Black box approach**: "It works! 2+2=4. Good enough."
- **Mech interp approach**: "Let me open it up. There are gears here... this one carries the tens digit... this spring triggers the display..."

Or think about a chess grandmaster:
- **Explainability**: "I moved there because it felt right"
- **Mech interp**: "I computed that after 5 moves, I'd have a discovered attack on the queen, which I noticed because I recognized a pattern from a famous game..."

We want to understand the ACTUAL computations, not just the outcomes.

### Why This Matters on DGX Spark

With DGX Spark's 128GB memory, you can analyze medium-sized language models (GPT-2 XL, Pythia-2.8B) and cache all their activations for deep analysis - something impossible on smaller hardware.

### When You're Ready for Details

→ See: Notebook 01, Section "What is mechanistic interpretability?"

---

## 🧒 The Residual Stream: The Model's Scratchpad

### The Jargon-Free Version

Information in a transformer flows through a "stream" that every layer can read from and write to. It's like a shared notebook that everyone contributes to.

### The Analogy

**The residual stream is like a collaborative Google Doc...**

Imagine a team writing a document together:
- **Editor 1 (Layer 1)**: Reads the doc, adds a paragraph about characters
- **Editor 2 (Layer 2)**: Reads everything so far, adds a paragraph about plot
- **Editor 3 (Layer 3)**: Reads it all, adds conclusions
- ...
- **Final editor**: Reads the completed doc, writes the answer

Each editor:
- Can see EVERYTHING written before them
- ADDS to the document (doesn't replace)
- Contributes their specialty

This is exactly how transformers work! Each layer reads the "residual stream" and adds its contribution.

### A Visual

```
The Residual Stream:

Input  →  Layer 1 adds  →  Layer 2 adds  →  Layer 3 adds  →  Output
"The"     + grammar       + meaning       + context          "cat"
          info            info           info

Everything accumulates! Layer 3 sees what Layers 1 & 2 wrote.
```

### When You're Ready for Details

→ See: Notebook 01, Section "The Residual Stream View"

---

## 🧒 Attention: The Model's Highlighter

### The Jargon-Free Version

Attention is how the model decides which words to look at when making predictions. Different "heads" look for different things, like multiple people reading the same text for different purposes.

### The Analogy

**Attention heads are like different readers with different highlighters...**

Imagine a sentence: "The cat sat on the mat because it was tired."

- **Reader 1 (grammar checker)**: Highlights "cat" when looking at "sat" (subject-verb agreement)
- **Reader 2 (reference resolver)**: Highlights "cat" when looking at "it" (what does "it" mean?)
- **Reader 3 (position tracker)**: Highlights the previous word, always

Each reader:
- Has ONE highlighter (can only mark, not change text)
- Has their own specialty
- Reports what they found to the team

Transformers have many "attention heads" - each is like a specialized reader looking for specific patterns!

### Common Misconception

❌ **People often think**: Attention shows what the model thinks is "important"
✅ **But actually**: Attention shows what each head LOOKS AT. High attention doesn't mean high importance for the final answer!

### When You're Ready for Details

→ See: Notebook 01, Section "Attention Head Roles"

---

## 🧒 Induction Heads: Copy-Paste Circuits

### The Jargon-Free Version

Induction heads are specific circuits that do in-context learning by completing patterns. When they see "Harry Potter... Harry", they predict "Potter" because they saw that pattern earlier.

### The Analogy

**Induction heads are like autocomplete on your phone...**

When you type "Happy birth", your phone suggests "day" because:
1. It remembers you've typed "Happy birthday" before
2. It recognizes you're in the same pattern again

Induction heads work similarly:
1. They notice "Harry" appeared before
2. They look at what came after "Harry" (→ "Potter")
3. They copy that to complete the pattern

```
"Harry Potter went to... Harry ___"

Induction head thinks:
1. "I've seen 'Harry' before..."
2. "After 'Harry' came 'Potter'..."
3. "I'll predict 'Potter'!"
```

### Why This Matters

Induction heads explain HOW models do in-context learning! They're not magic - they're pattern-completion circuits we can understand.

### When You're Ready for Details

→ See: Notebook 03, Section "Finding Induction Heads"

---

## 🧒 Activation Patching: Playing "What If?"

### The Jargon-Free Version

Activation patching means running the model twice (with different inputs), then swapping internal computations between runs to see what actually matters.

### The Analogy

**Activation patching is like swapping organs between identical twins...**

Imagine two identical twins thinking about different things:
- **Twin A**: Thinking about Paris
- **Twin B**: Thinking about London

To find which part of the brain handles cities:
1. Run both twins' thoughts
2. Copy Twin B's "city-processing" brain region into Twin A
3. Now Twin A thinks of... London!

This proves that brain region processes city information.

In transformers:
1. Run model on "The capital of France is" (→ Paris)
2. Run model on "The capital of England is" (→ London)
3. Copy specific activations from run 2 into run 1
4. See if the answer changes from Paris to London!

### A Visual

```
Run 1: "capital of France" → [Layer 5] → Paris
Run 2: "capital of England" → [Layer 5] → London

Patched run: "capital of France" → [COPY Layer 5 from Run 2] → London!

This proves: Layer 5 contains the country→capital mapping.
```

### When You're Ready for Details

→ See: Notebook 02, Section "Activation Patching"

---

## 🧒 Circuits: The Model's Algorithms

### The Jargon-Free Version

A circuit is a specific set of components (attention heads, MLP layers) that work together to implement a specific behavior. It's like finding the actual code the model runs.

### The Analogy

**Circuits are like finding the recipe in a restaurant's kitchen...**

You eat amazing pasta at a restaurant. You could:
1. **Explainability approach**: Ask the waiter "Why is this good?" ("The chef is talented!")
2. **Mech interp approach**: Go into the kitchen and write down every step the chef takes

The circuit IS the recipe:
- Step 1: Identify the subject (attention head 5.1)
- Step 2: Look up what comes after it (attention head 7.3)
- Step 3: Move that information to output (attention head 9.6)

**We can write down exactly what the model does!**

### Example: The IOI Circuit

For "John and Mary went to the store. John gave a book to [Mary]":

```
Circuit:
1. "Previous token heads" track positions
2. "Duplicate token heads" notice "John" appears twice
3. "S-inhibition heads" suppress "John" (already used)
4. "Name mover heads" copy "Mary" to the answer

Not magic - just specific, understandable steps!
```

### When You're Ready for Details

→ See: Notebook 02, Section "The IOI Circuit"

---

## 🧒 Superposition: Too Many Features, Not Enough Neurons

### The Jargon-Free Version

Models store more concepts than they have neurons by overlapping them. It's like a library storing more books than it has shelves by clever stacking.

### The Analogy

**Superposition is like a crowded coat check...**

Imagine a coat check with 100 hooks, but 1000 guests:
- **Naive approach**: Turn away 900 guests (not enough hooks!)
- **Superposition approach**: Stack coats carefully, use partial hooks

The model does this with concepts:
- 768 neurons, but 10,000+ concepts to store
- Each concept uses a DIRECTION in 768-dimensional space
- Directions can be almost-overlapping if features rarely co-occur

**Trade-off**: Occasional confusion when overlapping concepts both activate.

### Why This Matters

This is why interpretability is hard! A single neuron might represent multiple concepts, depending on context.

### When You're Ready for Details

→ See: Notebook 04, Section "Superposition and Features"

---

## 🧒 Sparse Autoencoders: Untangling the Mess

### The Jargon-Free Version

Sparse autoencoders (SAEs) are tools that try to separate the overlapping concepts in superposition into individual, interpretable features.

### The Analogy

**SAEs are like audio unmixing...**

You have a recording of a party where everyone talked at once:
- The recording = model activations (everything mixed together)
- SAE = software that separates into individual voices
- Output = clear, individual features (concepts)

Or think of unmixing colors:
- You see purple (mixed)
- SAE separates into "this much red" + "this much blue"
- Now you can study each color separately

### When You're Ready for Details

→ See: Notebook 04, Section "Sparse Autoencoders"

---

## 🔗 From ELI5 to Technical

| ELI5 Term | Technical Term | Where to Learn More |
|-----------|----------------|---------------------|
| "Looking inside" | Mechanistic Interpretability | Notebook 01 |
| "Scratchpad" | Residual Stream | Notebook 01 |
| "Highlighter" | Attention Head | Notebook 01 |
| "Copy-paste" | Induction Head | Notebook 03 |
| "Playing what-if" | Activation Patching | Notebook 02 |
| "The recipe" | Circuit | Notebook 02 |
| "Overlapping storage" | Superposition | Notebook 04 |
| "Unmixing" | Sparse Autoencoder | Notebook 04 |

---

## 💡 The "Explain It Back" Test

You truly understand these concepts when you can explain them to someone else without using jargon. Try explaining:

1. Why looking at attention patterns alone doesn't tell you what's important
2. How the model can "remember" patterns it saw earlier in the text
3. Why we do activation patching instead of just looking at which neurons fire
4. Why a neuron might represent multiple different concepts
