import gradio as gr
import transformers_gradio

gr.load(
    name='google/paligemma2-10b-ft-docci-448',
    src=transformers_gradio.registry,
).launch()