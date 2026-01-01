# Module 2.3: NLP & Transformers - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, try these:**

1. Check transformers version: `pip show transformers`
2. Verify model loading: Test with smaller model first
3. Check GPU memory: `nvidia-smi`
4. Clear cache: `torch.cuda.empty_cache()`

---

## Attention Implementation Errors

### Error: Attention shapes don't match

**Symptoms**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (X and Y)
```

**Cause**: Dimension mismatch in Q, K, V matrices.

**Solution**:
```python
# Check shapes at each step
print(f"Q shape: {Q.shape}")  # (batch, heads, seq_len, d_k)
print(f"K shape: {K.shape}")  # (batch, heads, seq_len, d_k)
print(f"V shape: {V.shape}")  # (batch, heads, seq_len, d_v)

# K^T should be (batch, heads, d_k, seq_len)
K_T = K.transpose(-2, -1)
print(f"K^T shape: {K_T.shape}")

# Scores should be (batch, heads, seq_len, seq_len)
scores = Q @ K_T
print(f"Scores shape: {scores.shape}")
```

**Common fixes**:
```python
# Wrong: Using last dim instead of second-to-last
K.transpose(-1, -2)  # Wrong!
K.transpose(-2, -1)  # Correct!

# Wrong: Forgetting to reshape for multi-head
Q = self.W_q(x)  # (batch, seq_len, d_model)
# Need to reshape to (batch, seq_len, num_heads, d_k)
Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
# Then transpose to (batch, num_heads, seq_len, d_k)
Q = Q.transpose(1, 2)
```

---

### Error: Attention weights don't sum to 1

**Symptoms**: Model not learning, attention visualization looks wrong.

**Cause**: Softmax applied to wrong dimension.

**Solution**:
```python
# Softmax should be over keys (last dimension)
attention_weights = F.softmax(scores, dim=-1)  # Correct!

# NOT over queries
# attention_weights = F.softmax(scores, dim=-2)  # Wrong!

# Verify
print(f"Sum along last dim: {attention_weights.sum(dim=-1)}")  # Should be all 1s
```

---

### Error: Causal mask not working

**Symptoms**: Decoder can "see the future" during training.

**Solution**:
```python
def create_causal_mask(seq_len, device='cuda'):
    """Future positions should be masked (set to -inf before softmax)"""
    # Create upper triangular matrix of 1s
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Convert to -inf where mask is 1
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Apply BEFORE softmax
scores = scores + causal_mask  # Add -inf to future positions
attention_weights = F.softmax(scores, dim=-1)  # -inf becomes 0
```

---

## Transformer Block Errors

### Error: Pre-LN vs Post-LN confusion

**Symptoms**: Training instability, NaN losses.

**Solutions**:
```python
# Post-LN (original transformer, less stable)
class PostLNBlock(nn.Module):
    def forward(self, x):
        x = self.norm1(x + self.attention(x))  # Norm after add
        x = self.norm2(x + self.ffn(x))
        return x

# Pre-LN (recommended, more stable)
class PreLNBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))  # Norm before attention
        x = x + self.ffn(self.norm2(x))
        return x
```

---

### Error: Gradients vanishing in deep transformer

**Symptoms**: Loss not decreasing, gradients near zero.

**Solutions**:
```python
# Solution 1: Use Pre-LN (more stable gradients)

# Solution 2: Proper initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_weights)

# Solution 3: Learning rate warmup
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

---

## Positional Encoding Errors

### Error: Positional encoding has wrong shape

**Symptoms**: Broadcasting error when adding to embeddings.

**Solution**:
```python
# Input: (batch, seq_len, d_model)
# PE should be (1, max_len, d_model) for broadcasting

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        # ... compute pe ...
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # pe[:, :x.size(1)]: (1, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]  # Broadcasting works!
```

---

### Error: Model doesn't learn word order

**Symptoms**: "dog bites man" gives same output as "man bites dog".

**Causes**:
1. Positional encoding not added
2. PE added after attention (wrong!)
3. PE values too small

**Solution**:
```python
# Correct order: Embed → Add PE → Attention
x = self.embedding(tokens)  # (batch, seq_len, d_model)
x = self.positional_encoding(x)  # Add position info
x = self.transformer(x)  # Now attention sees positions
```

---

## Tokenizer Errors

### Error: `KeyError: Token not in vocabulary`

**Symptoms**: Unknown tokens during tokenization.

**Solution**:
```python
# Use HuggingFace tokenizer (handles unknown tokens)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.encode("Hello, world!")  # Handles any input

# For custom tokenizer, add [UNK] token
vocab["[UNK]"] = len(vocab)
```

---

### Error: BPE merges not applying correctly

**Symptoms**: Tokenizer output is just characters.

**Solution**:
```python
def apply_bpe(word, merges):
    """Apply BPE merges in order"""
    tokens = list(word) + ['</w>']  # Add end-of-word marker

    while True:
        # Find pairs that exist in merges
        pairs = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in merges:
                pairs.append((merges[pair], pair, i))  # (priority, pair, index)

        if not pairs:
            break

        # Apply highest priority merge (lowest index = learned first)
        pairs.sort()
        _, best_pair, idx = pairs[0]
        merged = ''.join(best_pair)
        tokens = tokens[:idx] + [merged] + tokens[idx+2:]

    return tokens
```

---

## BERT Fine-tuning Errors

### Error: BERT accuracy stuck at 50% (binary classification)

**Causes**:
1. Learning rate too high
2. Not loading pre-trained weights
3. Wrong pooling strategy

**Solutions**:
```python
# Solution 1: Use recommended learning rates
# BERT authors recommend: 2e-5, 3e-5, 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Solution 2: Verify pre-trained weights are loaded
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
# Check that encoder weights are NOT random
print(model.bert.encoder.layer[0].attention.self.query.weight[:2, :5])

# Solution 3: Use [CLS] token correctly
outputs = model(input_ids, attention_mask)
# outputs.logits uses [CLS] representation by default
```

---

### Error: `IndexError: index out of range in self`

**Cause**: Input longer than model's max length (512 for BERT).

**Solution**:
```python
# Truncate inputs
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(
    text,
    truncation=True,      # Truncate to max_length
    max_length=512,       # BERT's limit
    padding='max_length',
    return_tensors='pt'
)
```

---

## Generation Errors

### Error: Generated text is repetitive

**Symptoms**: "The cat sat on the mat on the mat on the mat..."

**Solutions**:
```python
# Solution 1: Increase temperature
output = model.generate(
    input_ids,
    temperature=0.8,  # Higher = more random (default 1.0)
    max_new_tokens=100
)

# Solution 2: Use top-p (nucleus) sampling
output = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,  # Sample from top 90% probability mass
    max_new_tokens=100
)

# Solution 3: Add repetition penalty
output = model.generate(
    input_ids,
    repetition_penalty=1.2,  # Penalize repeated tokens
    max_new_tokens=100
)

# Solution 4: Use top-k sampling
output = model.generate(
    input_ids,
    do_sample=True,
    top_k=50,  # Sample from top 50 tokens
    max_new_tokens=100
)
```

---

### Error: Temperature = 0 causes error

**Cause**: Division by zero in softmax(logits / temperature).

**Solution**:
```python
# Don't use temperature=0
# Instead, use greedy decoding:
output = model.generate(
    input_ids,
    do_sample=False,  # Greedy = argmax
    max_new_tokens=100
)

# Or use very small temperature
temperature = 0.01  # Close to greedy but not exactly
```

---

## Reset Procedures

### Memory Reset

```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# Delete large models explicitly
del model
del tokenizer
torch.cuda.empty_cache()
```

### Model Cache Reset

```bash
# Clear HuggingFace cache (if corrupt)
rm -rf ~/.cache/huggingface/hub/models--bert-base-uncased
# Model will re-download on next load
```

---

## ❓ Frequently Asked Questions

### General Questions

**Q: Why do we need attention when RNNs already work?**

RNNs process sequences one step at a time, creating a bottleneck where all information must pass through a fixed-size hidden state. By the time an RNN reaches the end of a long sentence, it may have "forgotten" the beginning. Attention lets every position directly access every other position in parallel, enabling:
- Long-range dependency modeling
- Parallel processing (faster training on GPUs)
- Explicit interpretability (you can visualize what the model attends to)

---

**Q: What's the difference between self-attention and cross-attention?**

- **Self-attention**: Query, Key, and Value all come from the same sequence. Used in BERT encoders and GPT decoders.
- **Cross-attention**: Query comes from one sequence, Key/Value from another. Used in encoder-decoder models (like T5) where the decoder attends to encoder outputs.

---

**Q: How is BERT different from GPT?**

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional (sees all tokens) | Causal (sees only past tokens) |
| Pre-training | Masked Language Modeling | Next Token Prediction |
| Best for | Understanding (classification, NER, Q&A) | Generation (text completion, chat) |

---

### Attention Questions

**Q: Why do we scale by sqrt(d_k) in attention?**

Without scaling, as the dimension `d_k` increases, the dot products between queries and keys grow larger (variance = d_k). Large dot products push softmax into regions with very small gradients, making training difficult. Dividing by `sqrt(d_k)` normalizes the variance to approximately 1, keeping softmax in its "learning-friendly" region.

---

**Q: Why multiple attention heads instead of one large one?**

Multiple heads let the model jointly attend to information from different representation subspaces at different positions:
- One head might focus on syntax ("subject-verb agreement")
- Another on semantics ("what does 'it' refer to?")
- Another on local patterns ("adjacent words")

Having 8 heads of dimension 64 is more expressive than one head of dimension 512, while using the same total parameters.

---

**Q: What does the attention pattern actually look like in trained models?**

In trained models, different heads show distinct patterns:
- **Diagonal patterns**: Attending to self or adjacent tokens
- **Vertical patterns**: Many tokens attending to one important token (like [CLS])
- **Block patterns**: Attending within sentence boundaries
- **Sparse patterns**: Attending to specific syntactic relationships

---

### Positional Encoding Questions

**Q: Why does attention need positional encoding?**

Attention is permutation-invariant by design - it treats tokens as a "bag of words." Without positional encoding, "dog bites man" and "man bites dog" would produce identical outputs. Positional encoding adds unique patterns to each position so the model can distinguish word order.

---

**Q: What's the difference between sinusoidal and learned positional encodings?**

- **Sinusoidal**: Fixed mathematical patterns (sin/cos at different frequencies). Can generalize to unseen sequence lengths.
- **Learned**: Each position has a trainable embedding vector. More flexible but limited to training length.

Modern LLMs (Llama, Mistral) use **RoPE** (Rotary Position Embeddings) which applies rotation to Q/K based on position, naturally encoding relative positions.

---

**Q: What is RoPE and why is it popular?**

Rotary Position Embeddings (RoPE) encode position by rotating the query and key vectors. Benefits:
- Naturally encodes relative positions (useful for understanding "2 words apart")
- Better length extrapolation than absolute position encodings
- Used in Llama, Mistral, and most modern LLMs

---

### Tokenization Questions

**Q: Why not just split on spaces and punctuation?**

Word-level tokenization has problems:
1. **Vocabulary explosion**: English has millions of words
2. **Unknown words**: "ChatGPT" wasn't in training data
3. **Morphology lost**: "running", "runner", "run" are separate tokens despite sharing meaning

Subword tokenization (BPE, SentencePiece) balances vocabulary size with sequence length and handles unknown words gracefully.

---

**Q: What vocabulary size should I use?**

Common choices:
- **32K-50K**: Standard for most models (GPT-2, BERT, Llama)
- **100K+**: For multilingual models (handles more scripts)
- **Smaller (8K-16K)**: For specialized domains or smaller models

Larger vocabulary = shorter sequences but larger embedding matrix.

---

**Q: Why does the same text have different token counts in different models?**

Each model trains its own tokenizer on different data, resulting in different merge rules. "ChatGPT" might be 1 token in GPT-4 but 3 tokens in an older model that never saw this word during tokenizer training.

---

### Fine-tuning Questions

**Q: What learning rate should I use for BERT fine-tuning?**

The BERT authors recommend: `2e-5`, `3e-5`, or `5e-5`. This is much smaller than typical deep learning rates (0.001) because:
- BERT is already well-initialized
- Too high a learning rate destroys pre-trained knowledge
- Fine-tuning is subtle adjustment, not learning from scratch

---

**Q: Should I freeze BERT layers during fine-tuning?**

It depends:
- **Full fine-tuning** (unfreeze all): Best accuracy, but needs more compute
- **Freeze BERT, train classifier only**: Faster, good for small datasets
- **Gradual unfreezing**: Start frozen, unfreeze layers progressively

On DGX Spark with 128GB unified memory, you can typically do full fine-tuning efficiently.

---

**Q: My BERT accuracy is stuck at 50% for binary classification. What's wrong?**

Common issues:
1. **Learning rate too high**: Try 2e-5 instead of 1e-3
2. **Not loading pre-trained weights**: Verify with `model.bert.encoder.layer[0].attention.self.query.weight[:2, :2]` (should not be random)
3. **Wrong pooling**: Use the [CLS] token output, not averaging all tokens
4. **Data issue**: Check class balance and label encoding

---

### Generation Questions

**Q: What's the difference between temperature, top-k, and top-p?**

- **Temperature**: Scales logits before softmax. Low = confident/deterministic, high = random/creative
- **Top-k**: Only sample from the k most likely tokens
- **Top-p (nucleus)**: Sample from smallest set of tokens whose cumulative probability >= p

In practice, **top-p with temperature** is the most common combination (e.g., p=0.9, temp=0.8).

---

**Q: Why is my generated text repetitive?**

Greedy decoding (always picking the highest probability token) leads to repetition. Solutions:
1. Use sampling (`do_sample=True`)
2. Add temperature (0.7-0.9)
3. Use top-p sampling (p=0.9)
4. Add repetition penalty (1.1-1.3)

---

**Q: What is beam search and when should I use it?**

Beam search maintains multiple candidate sequences at each step, selecting the overall most likely sequence. Best for:
- Translation (need coherent output)
- Summarization (quality over creativity)
- Formal text generation

Avoid for creative writing where sampling provides better diversity.

---

### DGX Spark-Specific Questions

**Q: How does DGX Spark's 128GB unified memory help with transformers?**

Benefits include:
- **Larger batch sizes**: Use 64+ instead of typical 16-32
- **Longer sequences**: Process 8K+ tokens without memory issues
- **Bigger models**: Fine-tune models up to 12-16B parameters (full FP16)
- **No gradient checkpointing needed**: Keep all activations in memory

---

**Q: Which dtype should I use on DGX Spark?**

Use **bfloat16** (`torch.bfloat16`) for optimal performance on the Blackwell architecture. It provides:
- Native hardware acceleration
- Good numerical stability
- Half the memory of float32

```python
model = model.to(torch.bfloat16)
```

---

**Q: Why can't I pip install PyTorch on DGX Spark?**

DGX Spark uses ARM64/aarch64 architecture. Pre-built pip wheels are typically for x86. Use the NGC container which includes pre-built PyTorch optimized for the platform:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

### Debugging Questions

**Q: My attention shapes don't match. How do I debug?**

Print shapes at each step:
```python
print(f"Q: {Q.shape}")  # (batch, heads, seq_len, d_k)
print(f"K: {K.shape}")  # (batch, heads, seq_len, d_k)
print(f"K^T: {K.transpose(-2, -1).shape}")  # (batch, heads, d_k, seq_len)
print(f"scores: {(Q @ K.transpose(-2, -1)).shape}")  # (batch, heads, seq_len, seq_len)
```

Common issues:
- Forgetting to transpose K (`K.transpose(-2, -1)` not `K.transpose(-1, -2)`)
- Not reshaping for multi-head (`view` then `transpose`)
- Missing `.contiguous()` before `.view()`

---

**Q: I'm getting NaN losses during training. What's wrong?**

Common causes:
1. **Learning rate too high**: Reduce by 10x
2. **No gradient clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. **Missing warmup**: Use learning rate warmup for first ~10% of steps
4. **Numerical instability**: Check for divide-by-zero in attention (masked positions)

---

## Still Stuck?

1. **Print tensor shapes** - Most bugs are shape mismatches
2. **Check solution notebooks** - in `solutions/` folder
3. **Use smaller examples** - Debug with seq_len=3 before scaling up
4. **Visualize attention** - Often reveals what's wrong
5. **Review FAQ above** - For conceptual questions about transformers

**Debug template**:
```python
def debug_attention(Q, K, V, name=""):
    print(f"\n=== {name} ===")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    scores = Q @ K.transpose(-2, -1)
    print(f"Scores: {scores.shape}")
    print(f"Scores range: [{scores.min():.2f}, {scores.max():.2f}]")
    attn = F.softmax(scores, dim=-1)
    print(f"Attention sum: {attn.sum(dim=-1)[0, 0]}")  # Should be 1
```
