import gradio as gr
import transformers_gradio

gr.load(
    name='bartowski/Marco-o1-GGUF',
    src=transformers_gradio.registry,
).launch()