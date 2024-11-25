# `transformers-gradio`

is a Python package that makes it easy for developers to create machine learning apps powered by 🤗 Transformers models using Gradio.

# Installation

You can install `transformers-gradio` directly using pip:

```bash
pip install transformers-gradio
```

# Basic Usage

First, you'll need a Hugging Face model. Then in a Python file, write:

```python
import gradio as gr
from transformers_gradio import registry

interface = registry(
    model_path='organization/model-name'  # Hugging Face model ID
)
interface.launch()
```

Run the Python file, and you should see a Gradio chat interface connected to your local 🤗 Transformers model!

# Customization 

The interface includes several parameters that can be adjusted through the "⚙️ Parameters" accordion:

- System prompt (default: "You are a helpful AI assistant.")
- Temperature (0-1, default: 0.7)
- Max new tokens (128-4096, default: 1024)
- Top K sampling (1-80, default: 40)
- Repetition penalty (0-2, default: 1.1)
- Top P sampling (0-1, default: 0.95)

# Under the Hood

The library uses 🤗 Transformers with the following features:
- 4-bit quantization using BitsAndBytes (bfloat16)
- Automatic Flash Attention 2 installation with fallback to standard attention
- Streaming token generation using TextIteratorStreamer
- ChatML formatting for conversations
- Support for both text and image inputs (for multimodal models)
- Automatic device selection (CUDA if available, otherwise CPU)

-------

Note: Make sure you have a compatible GGUF model file before running the interface. You can download models from sources like Hugging Face or convert existing models to GGUF format.