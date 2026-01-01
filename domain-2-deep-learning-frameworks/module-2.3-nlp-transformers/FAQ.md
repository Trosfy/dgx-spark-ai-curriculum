# Module 2.3: NLP & Transformers - Frequently Asked Questions

## General Questions

### Q: Why do we need attention when RNNs already work?

**A:** RNNs process sequences one step at a time, creating a bottleneck where all information must pass through a fixed-size hidden state. By the time an RNN reaches the end of a long sentence, it may have "forgotten" the beginning. Attention lets every position directly access every other position in parallel, enabling:
- Long-range dependency modeling
- Parallel processing (faster training on GPUs)
- Explicit interpretability (you can visualize what the model attends to)

### Q: What's the difference between self-attention and cross-attention?

**A:**
- **Self-attention**: Query, Key, and Value all come from the same sequence. Used in BERT encoders and GPT decoders.
- **Cross-attention**: Query comes from one sequence, Key/Value from another. Used in encoder-decoder models (like T5) where the decoder attends to encoder outputs.

### Q: How is BERT different from GPT?

**A:**
| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Attention | Bidirectional (sees all tokens) | Causal (sees only past tokens) |
| Pre-training | Masked Language Modeling | Next Token Prediction |
| Best for | Understanding (classification, NER, Q&A) | Generation (text completion, chat) |

---

## Attention Questions

### Q: Why do we scale by sqrt(d_k) in attention?

**A:** Without scaling, as the dimension `d_k` increases, the dot products between queries and keys grow larger (variance = d_k). Large dot products push softmax into regions with very small gradients, making training difficult. Dividing by `sqrt(d_k)` normalizes the variance to approximately 1, keeping softmax in its "learning-friendly" region.

### Q: Why multiple attention heads instead of one large one?

**A:** Multiple heads let the model jointly attend to information from different representation subspaces at different positions:
- One head might focus on syntax ("subject-verb agreement")
- Another on semantics ("what does 'it' refer to?")
- Another on local patterns ("adjacent words")

Having 8 heads of dimension 64 is more expressive than one head of dimension 512, while using the same total parameters.

### Q: What does the attention pattern actually look like in trained models?

**A:** In trained models, different heads show distinct patterns:
- **Diagonal patterns**: Attending to self or adjacent tokens
- **Vertical patterns**: Many tokens attending to one important token (like [CLS])
- **Block patterns**: Attending within sentence boundaries
- **Sparse patterns**: Attending to specific syntactic relationships

---

## Positional Encoding Questions

### Q: Why does attention need positional encoding?

**A:** Attention is permutation-invariant by design - it treats tokens as a "bag of words." Without positional encoding, "dog bites man" and "man bites dog" would produce identical outputs. Positional encoding adds unique patterns to each position so the model can distinguish word order.

### Q: What's the difference between sinusoidal and learned positional encodings?

**A:**
- **Sinusoidal**: Fixed mathematical patterns (sin/cos at different frequencies). Can generalize to unseen sequence lengths.
- **Learned**: Each position has a trainable embedding vector. More flexible but limited to training length.

Modern LLMs (Llama, Mistral) use **RoPE** (Rotary Position Embeddings) which applies rotation to Q/K based on position, naturally encoding relative positions.

### Q: What is RoPE and why is it popular?

**A:** Rotary Position Embeddings (RoPE) encode position by rotating the query and key vectors. Benefits:
- Naturally encodes relative positions (useful for understanding "2 words apart")
- Better length extrapolation than absolute position encodings
- Used in Llama, Mistral, and most modern LLMs

---

## Tokenization Questions

### Q: Why not just split on spaces and punctuation?

**A:** Word-level tokenization has problems:
1. **Vocabulary explosion**: English has millions of words
2. **Unknown words**: "ChatGPT" wasn't in training data
3. **Morphology lost**: "running", "runner", "run" are separate tokens despite sharing meaning

Subword tokenization (BPE, SentencePiece) balances vocabulary size with sequence length and handles unknown words gracefully.

### Q: What vocabulary size should I use?

**A:** Common choices:
- **32K-50K**: Standard for most models (GPT-2, BERT, Llama)
- **100K+**: For multilingual models (handles more scripts)
- **Smaller (8K-16K)**: For specialized domains or smaller models

Larger vocabulary = shorter sequences but larger embedding matrix.

### Q: Why does the same text have different token counts in different models?

**A:** Each model trains its own tokenizer on different data, resulting in different merge rules. "ChatGPT" might be 1 token in GPT-4 but 3 tokens in an older model that never saw this word during tokenizer training.

---

## Fine-tuning Questions

### Q: What learning rate should I use for BERT fine-tuning?

**A:** The BERT authors recommend: `2e-5`, `3e-5`, or `5e-5`. This is much smaller than typical deep learning rates (0.001) because:
- BERT is already well-initialized
- Too high a learning rate destroys pre-trained knowledge
- Fine-tuning is subtle adjustment, not learning from scratch

### Q: Should I freeze BERT layers during fine-tuning?

**A:** It depends:
- **Full fine-tuning** (unfreeze all): Best accuracy, but needs more compute
- **Freeze BERT, train classifier only**: Faster, good for small datasets
- **Gradual unfreezing**: Start frozen, unfreeze layers progressively

On DGX Spark with 128GB unified memory, you can typically do full fine-tuning efficiently.

### Q: My BERT accuracy is stuck at 50% for binary classification. What's wrong?

**A:** Common issues:
1. **Learning rate too high**: Try 2e-5 instead of 1e-3
2. **Not loading pre-trained weights**: Verify with `model.bert.encoder.layer[0].attention.self.query.weight[:2, :2]` (should not be random)
3. **Wrong pooling**: Use the [CLS] token output, not averaging all tokens
4. **Data issue**: Check class balance and label encoding

---

## Generation Questions

### Q: What's the difference between temperature, top-k, and top-p?

**A:**
- **Temperature**: Scales logits before softmax. Low = confident/deterministic, high = random/creative
- **Top-k**: Only sample from the k most likely tokens
- **Top-p (nucleus)**: Sample from smallest set of tokens whose cumulative probability >= p

In practice, **top-p with temperature** is the most common combination (e.g., p=0.9, temp=0.8).

### Q: Why is my generated text repetitive?

**A:** Greedy decoding (always picking the highest probability token) leads to repetition. Solutions:
1. Use sampling (`do_sample=True`)
2. Add temperature (0.7-0.9)
3. Use top-p sampling (p=0.9)
4. Add repetition penalty (1.1-1.3)

### Q: What is beam search and when should I use it?

**A:** Beam search maintains multiple candidate sequences at each step, selecting the overall most likely sequence. Best for:
- Translation (need coherent output)
- Summarization (quality over creativity)
- Formal text generation

Avoid for creative writing where sampling provides better diversity.

---

## DGX Spark-Specific Questions

### Q: How does DGX Spark's 128GB unified memory help with transformers?

**A:** Benefits include:
- **Larger batch sizes**: Use 64+ instead of typical 16-32
- **Longer sequences**: Process 8K+ tokens without memory issues
- **Bigger models**: Fine-tune models up to 12-16B parameters (full FP16)
- **No gradient checkpointing needed**: Keep all activations in memory

### Q: Which dtype should I use on DGX Spark?

**A:** Use **bfloat16** (`torch.bfloat16`) for optimal performance on the Blackwell architecture. It provides:
- Native hardware acceleration
- Good numerical stability
- Half the memory of float32

```python
model = model.to(torch.bfloat16)
```

### Q: Why can't I pip install PyTorch on DGX Spark?

**A:** DGX Spark uses ARM64/aarch64 architecture. Pre-built pip wheels are typically for x86. Use the NGC container which includes pre-built PyTorch optimized for the platform:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

## Debugging Questions

### Q: My attention shapes don't match. How do I debug?

**A:** Print shapes at each step:
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

### Q: I'm getting NaN losses during training. What's wrong?

**A:** Common causes:
1. **Learning rate too high**: Reduce by 10x
2. **No gradient clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. **Missing warmup**: Use learning rate warmup for first ~10% of steps
4. **Numerical instability**: Check for divide-by-zero in attention (masked positions)

---

## Further Resources

- See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed error solutions
- See [ELI5.md](./ELI5.md) for intuitive explanations
- See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for code snippets
