import os
import subprocess
from huggingface_hub import hf_hub_download, list_repo_files
import gradio as gr
from typing import Callable
import base64
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from threading import Thread
from transformers import TextIteratorStreamer
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

__version__ = "0.0.1"


def load_vision_model(model_path: str, device: str):
    """Load vision-based models."""
    if "paligemma" in model_path.lower():
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def load_text_model(model_path: str, device: str):
    """Load text-based models."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if "olmo" in model_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        except Exception as e:
            print(f"Flash Attention failed, falling back to default attention: {str(e)}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
    return model, tokenizer

def format_chat_history(message: str, history: list, system_prompt: str, model_path: str):
    """Format chat history based on model type."""
    if "tulu" in model_path.lower():
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        return messages
    elif "olmo" in model_path.lower():
        instruction = f"<|endoftext|><|user|>\n{system_prompt}\n<|assistant|>\n"
        for user_msg, assistant_msg in history:
            instruction += f"<|endoftext|><|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        instruction += f"<|endoftext|><|user|>\n{message}\n<|assistant|>\n"
        return instruction
    else:
        instruction = '<|im_start|>system\n' + system_prompt + '\n<|im_end|>\n'
        for user_msg, assistant_msg in history:
            instruction += f'<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n{assistant_msg}\n<|im_end|>\n'
        instruction += f'<|im_start|>user\n{message}\n<|im_end|>\n<|im_start|>assistant\n'
        return instruction

def get_fn(model_path: str, **model_kwargs):
    """Create a chat function with the specified model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(model_path)
    
    # Determine if the model is a vision model
    is_vision_model = (
        (hasattr(config, 'is_vision_model') and config.is_vision_model) or
        any(name in model_path.lower() for name in ["idefics", "smolvlm", "paligemma"])
    )
    
    if is_vision_model:
        model, processor = load_vision_model(model_path, device)
        tokenizer = processor
    else:
        model, tokenizer = load_text_model(model_path, device)

    def predict(
        message: str,
        history,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_k: int,
        repetition_penalty: float,
        top_p: float
    ):
        try:
            # Check model type
            is_tulu = "tulu" in model_path.lower()
            is_olmo = "olmo" in model_path.lower()
            is_smolvlm = "smolvlm" in model_path.lower()
            is_paligemma = "paligemma" in model_path.lower()
            
            if is_smolvlm or is_paligemma:
                if isinstance(message, dict) and message.get("files"):
                    img = Image.open(message["files"][-1]).convert("RGB")
                    if is_paligemma:
                        # PaliGemma specific processing
                        prompt = "<image>" + (message.get("text", "") or "caption en")
                        inputs = processor(prompt, img, return_tensors="pt").to(device)
                        
                        # Set up streamer
                        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
                        generation_kwargs = dict(
                            **inputs,
                            streamer=streamer,
                            do_sample=True,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens,
                            repetition_penalty=repetition_penalty,
                            top_p=top_p
                        )

                        # Generate in a thread
                        thread = Thread(target=model.generate, kwargs=generation_kwargs)
                        thread.start()

                        # Stream the output
                        response_text = ""
                        for new_text in streamer:
                            response_text += new_text
                            yield response_text.strip()
                
            else:
                # Check if using Tulu or OLMo model
                is_tulu = "tulu" in model_path.lower()
                is_olmo = "olmo" in model_path.lower()
                
                if is_tulu:
                    # Format conversation for Tulu models
                    messages = [{"role": "system", "content": system_prompt}]
                    for user_msg, assistant_msg in history:
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": assistant_msg})
                    messages.append({"role": "user", "content": message})
                    
                    # Convert messages to string format
                    instruction = tokenizer.apply_chat_template(messages, tokenize=False)
                elif is_olmo:
                    # Format conversation for OLMo models
                    instruction = f"<|endoftext|><|user|>\n{system_prompt}\n<|assistant|>\n"
                    for user_msg, assistant_msg in history:
                        instruction += f"<|endoftext|><|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
                    instruction += f"<|endoftext|><|user|>\n{message}\n<|assistant|>\n"
                else:
                    # Original ChatML format
                    instruction = '<|im_start|>system\n' + system_prompt + '\n<|im_end|>\n'
                    for user_msg, assistant_msg in history:
                        instruction += f'<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n{assistant_msg}\n<|im_end|>\n'
                    instruction += f'<|im_start|>user\n{message}\n<|im_end|>\n<|im_start|>assistant\n'

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
                input_ids, attention_mask = enc.input_ids, enc.attention_mask

                # Truncate if needed
                if input_ids.shape[1] > 8192:  # Using n_ctx from original
                    input_ids = input_ids[:, -8192:]
                    attention_mask = attention_mask[:, -8192:]

                generate_kwargs = dict(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    streamer=streamer,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p
                )

                t = Thread(target=model.generate, kwargs=generate_kwargs)
                t.start()

                response_text = ""
                for new_token in streamer:
                    if new_token in ["<|endoftext|>", "<|im_end|>"]:
                        break
                    response_text += new_token
                    yield response_text.strip()

                if not response_text.strip():
                    yield "I apologize, but I was unable to generate a response. Please try again."

            if not response_text.strip():
                yield "I apologize, but I was unable to generate a response. Please try again."

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    return predict


def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string


def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif", "pdf"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            messages = []
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    messages.append({"role": "user", "content": handle_user_msg(user_msg)})
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg
            if type(message) is str and files is not None:
                message = {"text":message, "files":files}
            elif type(message) is dict and files is not None:
                if message["files"] is None or len(message["files"]) == 0:
                    message["files"] = files
            messages.append({"role": "user", "content": handle_user_msg(message)})
            return {"messages": messages}

        postprocess = lambda x: x
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def get_model_path(name: str = None, model_path: str = None) -> str:
    """Get the local path to the model."""
    if model_path:
        return model_path
    
    if name:
        if "/" in name:
            return name
        else:
            model_mapping = {
                "tulu-3": "allenai/llama-tulu-3-8b",
                "olmo-2-13b": "allenai/OLMo-2-1124-13B-Instruct",
                "smolvlm": "HuggingFaceTB/SmolVLM-Instruct",  # Add SmolVLM mapping
                # ... other mappings ...
            }
            if name not in model_mapping:
                raise ValueError(f"Unknown model name: {name}")
            return model_mapping[name]
    
    raise ValueError("Either name or model_path must be provided")


def registry(name: str = None, model_path: str = None, **kwargs):
    """Create a Gradio Interface with similar styling and parameters."""
    
    model_path = get_model_path(name, model_path)
    fn = get_fn(model_path, **kwargs)

    interface = gr.ChatInterface(
        fn=fn,
        multimodal=True,  # Enable multimodal input
        additional_inputs_accordion=gr.Accordion("⚙️ Parameters", open=False),
        additional_inputs=[
            gr.Textbox(
                "You are a helpful AI assistant.",
                label="System prompt"
            ),
            gr.Slider(0, 1, 0.7, label="Temperature"),
            gr.Slider(128, 4096, 1024, label="Max new tokens"),
            gr.Slider(1, 80, 40, label="Top K sampling"),
            gr.Slider(0, 2, 1.1, label="Repetition penalty"),
            gr.Slider(0, 1, 0.95, label="Top P sampling"),
        ],
    )
    
    return interface