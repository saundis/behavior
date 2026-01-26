"""
Model loading utilities for behavior experiments.

Provides functions to load OLMo and other models with various configurations
including quantization for memory efficiency.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelPair:
    """Container for a pair of models used in interpolation experiments."""
    model_m: AutoModelForCausalLM
    model_n: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    name_m: str
    name_n: str

    def __repr__(self):
        return f"ModelPair(M={self.name_m}, N={self.name_n})"


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load tokenizer for a model.

    Args:
        model_name: HuggingFace model identifier (e.g., 'allenai/OLMo-1B')

    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set (some models don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name: str,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> AutoModelForCausalLM:
    """
    Load a causal language model with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', or 'auto')
        load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        torch_dtype: Data type for model weights (e.g., torch.float16)

    Returns:
        Loaded model
    """
    # Set default dtype for non-quantized models
    if torch_dtype is None and not (load_in_8bit or load_in_4bit):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Configure loading arguments
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "auto" else None,
    }

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Move to device if not using device_map
    if load_kwargs.get("device_map") is None:
        model = model.to(device)

    # Set to eval mode
    model.eval()

    return model


def load_model_pair(
    model_m_name: str,
    model_n_name: str,
    device: str = "cuda",
    load_in_8bit: bool = False,
    sequential: bool = True,
) -> ModelPair:
    """
    Load a pair of models for interpolation experiments.

    Args:
        model_m_name: Name of model M
        model_n_name: Name of model N
        device: Device to load models on
        load_in_8bit: Use 8-bit quantization
        sequential: If True, models share memory more efficiently

    Returns:
        ModelPair containing both models and shared tokenizer
    """
    # Load tokenizer (use model M's tokenizer, assuming compatible)
    tokenizer = load_tokenizer(model_m_name)

    # Load both models
    print(f"Loading model M: {model_m_name}")
    model_m = load_model(model_m_name, device=device, load_in_8bit=load_in_8bit)

    print(f"Loading model N: {model_n_name}")
    model_n = load_model(model_n_name, device=device, load_in_8bit=load_in_8bit)

    return ModelPair(
        model_m=model_m,
        model_n=model_n,
        tokenizer=tokenizer,
        name_m=model_m_name,
        name_n=model_n_name,
    )


def load_model_organisms_pair(
    device: str = "cuda",
    domain: str = "medical",
) -> ModelPair:
    """
    Load Model Organisms pair: base model and misaligned variant for interpolation.

    The Model Organisms for Emergent Misalignment project provides models with
    known, labeled behavioral differences. These are full fine-tunes (not LoRA)
    that exhibit specific misaligned behaviors.

    Available domains:
        - "medical": Bad medical advice variant
        - "financial": Risky financial advice variant
        - "sports": Extreme sports advice variant

    Args:
        device: Device to load models on ('cuda', 'cpu', or 'auto')
        domain: Which domain to use ("medical", "financial", or "sports")

    Returns:
        ModelPair where:
            - model_n (alpha=0): Base Qwen2.5-0.5B-Instruct with normal behavior
            - model_m (alpha=1): Misaligned fine-tune
    """
    # Map domain to model name
    domain_to_model = {
        "medical": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
        "financial": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
        "sports": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
    }

    if domain not in domain_to_model:
        raise ValueError(f"Unknown domain: {domain}. Choose from: {list(domain_to_model.keys())}")

    base_name = "Qwen/Qwen2.5-0.5B-Instruct"
    misaligned_name = domain_to_model[domain]

    print(f"Loading base model: {base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (Model N - normal behavior)
    model_n = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model_n = model_n.to(device)
    model_n.eval()

    print(f"Loading misaligned model: {misaligned_name}")
    # Load misaligned model (Model M - bad behavior)
    model_m = AutoModelForCausalLM.from_pretrained(
        misaligned_name,
        torch_dtype=torch.float16,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model_m = model_m.to(device)
    model_m.eval()

    print("Models loaded successfully!")
    return ModelPair(
        model_m=model_m,
        model_n=model_n,
        tokenizer=tokenizer,
        name_m=f"Qwen2.5-0.5B-{domain}-misaligned",
        name_n="Qwen2.5-0.5B-Instruct-base",
    )


def get_model_memory_usage(model: AutoModelForCausalLM) -> dict:
    """
    Get memory usage statistics for a model.

    Returns:
        Dictionary with memory usage info
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "memory_mb": "unknown"}

    # Get GPU memory usage
    memory_allocated = torch.cuda.memory_allocated() / 1024**2
    memory_reserved = torch.cuda.memory_reserved() / 1024**2

    return {
        "device": str(next(model.parameters()).device),
        "memory_allocated_mb": round(memory_allocated, 2),
        "memory_reserved_mb": round(memory_reserved, 2),
    }


# Pre-defined model configurations for easy access
OLMO_MODELS = {
    "olmo-1b": "allenai/OLMo-1B",
    "olmo-7b": "allenai/OLMo-7B",
    "olmo-7b-instruct": "allenai/OLMo-7B-Instruct",
    "olmo-7b-sft": "allenai/OLMo-7B-SFT",
}

# Alternative models that might be useful
PYTHIA_MODELS = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}
