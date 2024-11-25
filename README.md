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
from transformers_gradio import TransformersGradio

gr.load(
    model_path='organization/model-name',  # Hugging Face model ID
    src=TransformersGradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to your local 🤗 Transformers model!

# Customization 

The interface includes several parameters that can be adjusted through the UI:

- System prompt
- Temperature (0-1)
- Max new tokens (128-4096)
- Top K sampling (1-80)
- Repetition penalty (0-2)
- Top P sampling (0-1)

The model uses 4-bit quantization by default and will attempt to use Flash Attention 2 if available, falling back to standard attention if the installation fails.

```python
import gradio as gr
from transformers_gradio import TransformersGradio

gr.load(
    model_path='organization/model-name',
    src=TransformersGradio.registry,
    # Add any additional model kwargs here
).launch()
```

# Under the Hood

The library uses 🤗 Transformers with the following features:
- 4-bit quantization using bitsandbytes
- Automatic Flash Attention 2 installation attempt
- Streaming token generation
- ChatML formatting for conversations
- Support for both text and image inputs (for multimodal models)

-------

Note: Make sure you have a compatible GGUF model file before running the interface. You can download models from sources like Hugging Face or convert existing models to GGUF format.