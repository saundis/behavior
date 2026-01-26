"""
Logit extraction utilities.

Provides functions to extract next-token logits from language models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Union, List


@torch.no_grad()
def get_next_token_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Get the logits for the next token given a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        device: Device to run on (defaults to model's device)

    Returns:
        Tensor of shape (vocab_size,) containing logits for next token
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Get model output
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract logits for the last position (next token prediction)
    # Shape: (batch_size, seq_len, vocab_size) -> (vocab_size,)
    next_token_logits = outputs.logits[0, -1, :]

    return next_token_logits


@torch.no_grad()
def get_logits_for_sequence(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Get logits for all positions in a sequence.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input text prompt
        device: Device to run on

    Returns:
        Tensor of shape (seq_len, vocab_size) containing logits for each position
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shape: (batch_size, seq_len, vocab_size) -> (seq_len, vocab_size)
    return outputs.logits[0]


@torch.no_grad()
def get_logits_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: Optional[str] = None,
    padding: bool = True,
) -> torch.Tensor:
    """
    Get next-token logits for a batch of prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        device: Device to run on
        padding: Whether to pad sequences

    Returns:
        Tensor of shape (batch_size, vocab_size) containing logits for each prompt
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=padding,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get logits at the last non-padded position for each sequence
    batch_size = input_ids.shape[0]
    seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because 0-indexed

    next_token_logits = torch.stack([
        outputs.logits[i, seq_lengths[i], :]
        for i in range(batch_size)
    ])

    return next_token_logits


def logits_to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert logits to probabilities using softmax.

    Args:
        logits: Raw logits tensor
        temperature: Temperature for softmax (higher = more uniform)

    Returns:
        Probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    return torch.softmax(logits / temperature, dim=-1)


def get_top_tokens(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 10,
) -> List[tuple]:
    """
    Get the top-k tokens and their probabilities from logits.

    Args:
        logits: Logits tensor of shape (vocab_size,)
        tokenizer: Tokenizer for decoding
        k: Number of top tokens to return

    Returns:
        List of (token_str, probability) tuples
    """
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))

    return results


def compare_logit_distributions(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 20,
) -> dict:
    """
    Compare two logit distributions and find interesting differences.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        tokenizer: Tokenizer for decoding
        top_k: Number of tokens to analyze

    Returns:
        Dictionary with comparison statistics
    """
    probs_m = torch.softmax(logits_m, dim=-1)
    probs_n = torch.softmax(logits_n, dim=-1)

    # Find tokens with largest probability differences
    prob_diff = probs_m - probs_n
    top_diff_positive, top_idx_positive = torch.topk(prob_diff, top_k)
    top_diff_negative, top_idx_negative = torch.topk(-prob_diff, top_k)

    # Decode tokens
    more_likely_in_m = [
        (tokenizer.decode([idx]), prob_diff[idx].item(), probs_m[idx].item(), probs_n[idx].item())
        for idx in top_idx_positive.tolist()
    ]

    more_likely_in_n = [
        (tokenizer.decode([idx]), -prob_diff[idx].item(), probs_m[idx].item(), probs_n[idx].item())
        for idx in top_idx_negative.tolist()
    ]

    # KL divergence (M || N) with numerical stability
    eps = 1e-8
    probs_m_safe = torch.clamp(probs_m, min=eps)
    probs_n_safe = torch.clamp(probs_n, min=eps)
    kl_div = torch.sum(probs_m_safe * (torch.log(probs_m_safe) - torch.log(probs_n_safe)))

    return {
        "more_likely_in_m": more_likely_in_m,
        "more_likely_in_n": more_likely_in_n,
        "kl_divergence_m_n": kl_div.item(),
        "top_token_m": tokenizer.decode([probs_m.argmax().item()]),
        "top_token_n": tokenizer.decode([probs_n.argmax().item()]),
    }
