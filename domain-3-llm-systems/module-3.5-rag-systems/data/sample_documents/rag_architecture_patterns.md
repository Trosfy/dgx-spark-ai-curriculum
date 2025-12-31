# RAG Architecture Patterns and Best Practices

## Introduction

Retrieval-Augmented Generation (RAG) is a paradigm that enhances large language models by incorporating external knowledge retrieval. Instead of relying solely on the model's parametric memory (learned during training), RAG systems retrieve relevant information from a knowledge base and use it to generate more accurate, up-to-date, and verifiable responses.

## Why RAG?

### Limitations of Pure LLMs
Large language models face several challenges:

1. **Knowledge Cutoff**: Training data has a fixed date
2. **Hallucinations**: Models can generate plausible but false information
3. **No Source Attribution**: Difficult to verify claims
4. **Static Knowledge**: Can't incorporate new information without retraining
5. **Context Limits**: Can't memorize entire document collections

### RAG Advantages
RAG addresses these limitations:

- **Dynamic Knowledge**: Retrieve current information
- **Grounded Responses**: Answers based on actual documents
- **Source Citations**: Can link to original sources
- **Scalable Knowledge**: Add documents without retraining
- **Domain Expertise**: Incorporate specialized knowledge bases

## Basic RAG Architecture

### Core Components
A basic RAG system consists of:

1. **Document Loader**: Ingests raw documents
2. **Chunking**: Splits documents into manageable pieces
3. **Embedding Model**: Converts text to vectors
4. **Vector Store**: Stores and retrieves embeddings
5. **Retriever**: Finds relevant chunks
6. **LLM**: Generates responses using retrieved context

### Basic Pipeline Flow
```
User Query → Embed Query → Vector Search → Top-K Chunks →
→ Construct Prompt → LLM Generation → Response
```

## Chunking Strategies

### Fixed-Size Chunking
The simplest approach: split by token/character count.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Pros**: Simple, predictable chunk sizes
**Cons**: May split sentences mid-thought

### Semantic Chunking
Split by meaning boundaries rather than fixed size.

**Approaches**:
- Section headers and paragraph breaks
- Sentence boundary detection
- Topic modeling

### Sentence-Based Chunking
Group sentences into chunks.

```python
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=50,
    tokens_per_chunk=256
)
```

### Hierarchical Chunking
Create multiple granularities:
- Document → Section → Paragraph → Sentence
- Link chunks to their parents for context

### Chunk Size Considerations
| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (100-200 tokens) | Precise retrieval | May lack context |
| Medium (300-500 tokens) | Balanced | Standard choice |
| Large (500-1000 tokens) | Rich context | May include irrelevant info |

## Embedding Models

### Popular Options
| Model | Dimensions | Quality | Speed |
|-------|------------|---------|-------|
| BGE-large-en-v1.5 | 1024 | Excellent | Medium |
| E5-large-v2 | 1024 | Excellent | Medium |
| nomic-embed-text | 768 | Good | Fast |
| OpenAI ada-002 | 1536 | Good | API latency |

### Embedding Best Practices
1. **Match domains**: Use domain-specific embeddings when available
2. **Normalize**: L2-normalize embeddings for cosine similarity
3. **Prefix queries**: Some models expect prefixes (e.g., "query: " or "search_document: ")
4. **Batch processing**: Embed documents in batches for efficiency

## Vector Databases

### Options Comparison
| Database | Strengths | Best For |
|----------|-----------|----------|
| ChromaDB | Simple, Python-native | Development, small scale |
| FAISS | Fast, GPU support | Pure performance |
| Qdrant | Production features, filtering | Production deployments |
| Pinecone | Fully managed, scalable | Enterprise, managed service |
| Weaviate | GraphQL, hybrid search | Complex queries |
| Milvus | Scalable, cloud-native | Large-scale deployments |

### Index Types
1. **Flat (Exact)**: 100% recall, O(n) search
2. **IVF (Inverted File)**: Clustered search, fast, good recall
3. **HNSW**: Graph-based, fast, excellent recall
4. **LSH (Locality Sensitive Hashing)**: Approximate, memory-efficient

## Advanced Retrieval Techniques

### Hybrid Search
Combine dense (embedding) and sparse (keyword) retrieval:

```python
def hybrid_search(query, alpha=0.5):
    # Dense retrieval
    dense_results = vector_db.similarity_search(query, k=50)

    # Sparse retrieval (BM25)
    sparse_results = bm25_search(query, k=50)

    # Reciprocal Rank Fusion
    return reciprocal_rank_fusion(dense_results, sparse_results, alpha)
```

**When to use**:
- Technical documents with specific terminology
- Queries with important keywords
- Mixed exact-match and semantic queries

### Query Expansion
Improve retrieval by expanding the query:

1. **LLM-Based Rewriting**: Have the LLM rephrase the query
2. **Multi-Query**: Generate multiple query variations
3. **Synonym Expansion**: Add synonyms automatically

### Hypothetical Document Embedding (HyDE)
Generate a hypothetical answer, then search for similar real documents:

```python
def hyde_search(query):
    # Generate hypothetical answer
    hypothetical = llm.generate(f"Write a passage that answers: {query}")

    # Embed the hypothetical
    hyp_embedding = embedder.encode(hypothetical)

    # Search with hypothetical embedding
    return vector_db.similarity_search_by_vector(hyp_embedding)
```

### Reranking
Two-stage retrieval for better precision:

```python
def rerank_search(query, k=5, first_stage_k=50):
    # First stage: fast retrieval
    candidates = retriever.retrieve(query, k=first_stage_k)

    # Second stage: cross-encoder reranking
    pairs = [(query, doc.content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Return top-k after reranking
    return sorted(zip(candidates, scores), key=lambda x: -x[1])[:k]
```

## Prompt Engineering for RAG

### Basic RAG Prompt
```
Answer the question based only on the following context:

Context:
{retrieved_chunks}

Question: {user_question}

Answer:
```

### Enhanced RAG Prompt
```
You are a helpful assistant that answers questions based on provided documentation.

INSTRUCTIONS:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Cite your sources using [Source: document_name]
4. Be concise and accurate

CONTEXT:
{retrieved_chunks}

QUESTION: {user_question}

ANSWER (with citations):
```

### Handling No Results
```python
if len(retrieved_chunks) == 0 or max_similarity < threshold:
    return "I couldn't find relevant information in the knowledge base to answer this question."
```

## RAG Evaluation

### Retrieval Metrics
- **Recall@K**: Are relevant documents in top K?
- **MRR (Mean Reciprocal Rank)**: Where is the first relevant doc?
- **NDCG**: Graded relevance ranking quality

### Generation Metrics (RAGAS)
- **Faithfulness**: Is the answer grounded in retrieved docs?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are retrieved docs relevant?
- **Context Recall**: Are all needed docs retrieved?

### Human Evaluation
- Factual accuracy
- Completeness
- Usefulness
- Citation accuracy

## Production Considerations

### Caching
- Cache embedding computations
- Cache frequent query results
- Use semantic caching for similar queries

### Monitoring
- Track retrieval latency
- Monitor generation quality scores
- Alert on low-confidence responses

### Error Handling
- Handle empty retrieval results
- Manage rate limits on LLM calls
- Graceful degradation

### Security
- Sanitize user inputs
- Access control for documents
- Audit logging for compliance

## Anti-Patterns to Avoid

### Common Mistakes
1. **Too Small Chunks**: Losing context
2. **No Overlap**: Missing relevant content at boundaries
3. **No Metadata**: Losing source/date information
4. **Single Retrieval Method**: Missing relevant docs
5. **No Evaluation**: Not measuring quality

### Red Flags
- Hallucinations despite having RAG
- Low retrieval scores
- Users not finding information they know exists
- Inconsistent answers for similar queries

## Conclusion

RAG is a powerful paradigm for building practical LLM applications. Success requires careful attention to each component: chunking, embeddings, retrieval, and generation. Regular evaluation and iteration are essential for maintaining quality as the knowledge base grows.
