# Module 2.5: Hugging Face Ecosystem - Quickstart

## Time: ~5 minutes

## What You'll Do
Use the Pipeline API to run 5 different AI tasks with one line of code each.

## Before You Start
- [ ] DGX Spark container running
- [ ] transformers library installed

## Let's Go!

### Step 1: Text Generation
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", device=0)
output = generator("The future of AI is", max_new_tokens=30, do_sample=True)
print("Text Generation:")
print(output[0]['generated_text'])
print()
```

### Step 2: Sentiment Analysis
```python
classifier = pipeline("sentiment-analysis", device=0)
result = classifier("I absolutely love learning about deep learning on DGX Spark!")
print(f"Sentiment: {result[0]['label']} ({result[0]['score']:.2%})")
print()
```

### Step 3: Question Answering
```python
qa = pipeline("question-answering", device=0)
result = qa(
    question="What is DGX Spark?",
    context="DGX Spark is NVIDIA's personal AI supercomputer with 128GB unified memory and a Blackwell GPU."
)
print(f"Q: What is DGX Spark?")
print(f"A: {result['answer']} (confidence: {result['score']:.2%})")
print()
```

### Step 4: Named Entity Recognition
```python
ner = pipeline("ner", aggregation_strategy="simple", device=0)
result = ner("NVIDIA released the DGX Spark at CES 2025 in Las Vegas.")
print("Named Entities:")
for entity in result:
    print(f"  {entity['word']}: {entity['entity_group']}")
print()
```

### Step 5: Summarization
```python
summarizer = pipeline("summarization", device=0)
text = """
The DGX Spark is NVIDIA's first personal AI supercomputer. It features the Blackwell
GB10 Superchip with 128GB of unified LPDDR5X memory. This enables running models up to
200B parameters with NVFP4 quantization. The system is designed for AI developers,
researchers, and students who need professional AI capabilities at home.
"""
result = summarizer(text, max_length=50, min_length=20)
print("Summary:")
print(result[0]['summary_text'])
```

## You Did It!

You just used 5 different AI pipelines:
- **Text Generation** - GPT-style text completion
- **Sentiment Analysis** - Classify positive/negative
- **Question Answering** - Extract answers from context
- **Named Entity Recognition** - Find people, places, organizations
- **Summarization** - Condense long text

All with just `pipeline("task_name")`!

In the full module, you'll learn:
- Exploring the Hugging Face Hub
- Loading any model with AutoModel
- Processing datasets with the datasets library
- Fine-tuning with the Trainer API
- Parameter-efficient fine-tuning (LoRA/PEFT)

## Next Steps
1. **Explore models**: Browse [huggingface.co](https://huggingface.co/models)
2. **Try more pipelines**: `translation`, `fill-mask`, `text-classification`
3. **Full tutorial**: Start with [Lab 2.5.1](./labs/lab-2.5.1-hub-exploration.ipynb)
