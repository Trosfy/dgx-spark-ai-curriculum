# Example: Domain-Specific AI Assistant

This is a minimal working example for Option A: Domain-Specific AI Assistant.

## What This Example Includes

- `simple_assistant.py` - A basic assistant implementation
- `example_tools.py` - Example custom tools
- `demo.py` - Quick demonstration script

## Quick Start

```bash
# Navigate to this directory
cd examples/option-a-assistant

# Run the demo
python demo.py
```

## Files Overview

### simple_assistant.py

A minimal assistant implementation with:
- Basic RAG capability (mock for demo)
- Simple tool calling
- Streaming responses

### example_tools.py

Example domain tools including:
- `calculate` - Simple calculator
- `search_docs` - Mock document search
- `get_weather` - Mock weather lookup

### demo.py

Interactive demo showing:
- Question answering
- Tool usage
- Streaming output

## Extending This Example

To build your full capstone, you'll want to:

1. **Replace mock components** with real implementations
2. **Add your domain knowledge** to the RAG system
3. **Fine-tune the base model** with QLoRA
4. **Add real tools** for your domain
5. **Create a proper API** with FastAPI
6. **Build evaluation** benchmarks

## Memory Requirements

This example is designed to run on DGX Spark:
- Uses Llama 3.3 8B (smaller for demo)
- 4-bit quantization
- Estimated: ~6GB GPU memory

For your full capstone, you can scale up to 70B models.

## Next Steps

1. Review the code in each file
2. Run the demo to see it working
3. Modify for your specific domain
4. Follow the main Option A notebook for full implementation
