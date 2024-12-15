from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer: Uses Hugging Face's AutoTokenizer to load the tokenizer associated with the model, ensuring the padding side is set to "right" (commonly used for decoder-based models)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files: Safetensors is a secure and efficient file format for storing model weights.
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    # Loads the tensors from the safetensors files into memory. The safe_open method ensures compatibility with PyTorch and supports loading directly to the CPU or GPU
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config: Instantiates a PaliGemmaConfig object to initialize the model architecture.
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration and transfers it to the specified device (e.g., GPU or CPU).
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model: Loads the weights (state dictionary) from the tensors dictionary into the model. Setting strict=False allows for flexibility if certain keys in the state dict are missing or extra.
    model.load_state_dict(tensors, strict=False)

    # Tie weights: Ensures that tied embeddings (e.g., input and output embeddings) are shared for memory efficiency and better generalization.
    model.tie_weights()

    return (model, tokenizer)

# Use Case:
# This function is designed for workflows requiring multi-modal models that integrate text and vision. Typical applications include:
# - Loading pre-trained weights for inference (e.g., caption generation, question answering).
# - Initializing models for fine-tuning on specific tasks.