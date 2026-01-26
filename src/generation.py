"""
Text generation utilities using interpolated logits.

Provides functions to generate completions by sampling from interpolated
logit distributions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Tuple
from tqdm import tqdm

try:
    from .logits import get_next_token_logits
    from .interpolation import interpolate_logits
except ImportError:
    from logits import get_next_token_logits
    from interpolation import interpolate_logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Sample a token from a logit distribution.

    Args:
        logits: Logits tensor of shape (vocab_size,)
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling with this threshold

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy decoding
        return logits.argmax().item()

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Apply top-p (nucleus) filtering
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


def generate_from_interpolated_logits(
    model_m: AutoModelForCausalLM,
    model_n: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    alpha: float,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    stop_tokens: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Tuple[str, List[int]]:
    """
    Generate text by sampling from interpolated logit distributions.

    At each step:
    1. Get logits from both models for current sequence
    2. Interpolate: L_alpha = alpha * L_M + (1 - alpha) * L_N
    3. Sample next token from interpolated distribution
    4. Append to sequence and repeat

    Args:
        model_m: First model (M)
        model_n: Second model (N)
        tokenizer: Shared tokenizer
        prompt: Starting prompt
        alpha: Interpolation parameter
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        stop_tokens: List of strings that stop generation
        device: Device to use

    Returns:
        Tuple of (generated_text, list_of_token_ids)
    """
    if device is None:
        device = next(model_m.parameters()).device

    # Convert stop tokens to IDs
    stop_token_ids = set()
    if stop_tokens:
        for st in stop_tokens:
            ids = tokenizer.encode(st, add_special_tokens=False)
            stop_token_ids.update(ids)

    # Add EOS token as stop
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)

    # Initialize with prompt
    current_text = prompt
    generated_tokens = []

    for _ in range(max_new_tokens):
        # Get logits from both models
        logits_m = get_next_token_logits(model_m, tokenizer, current_text, device)
        logits_n = get_next_token_logits(model_n, tokenizer, current_text, device)

        # Interpolate
        interpolated_logits = interpolate_logits(logits_m, logits_n, alpha)

        # Sample next token
        next_token = sample_token(
            interpolated_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Check for stop
        if next_token in stop_token_ids:
            break

        # Append to sequence
        generated_tokens.append(next_token)
        current_text = current_text + tokenizer.decode([next_token])

    # Get just the generated part
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text, generated_tokens


def generate_samples(
    model_m: AutoModelForCausalLM,
    model_n: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    alpha: float,
    num_samples: int = 10,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> List[str]:
    """
    Generate multiple samples from interpolated distribution.

    Args:
        model_m: First model
        model_n: Second model
        tokenizer: Tokenizer
        prompt: Starting prompt
        alpha: Interpolation parameter
        num_samples: Number of samples to generate
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling
        show_progress: Show progress bar
        device: Device to use

    Returns:
        List of generated texts
    """
    samples = []
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating (alpha={alpha})")

    for _ in iterator:
        text, _ = generate_from_interpolated_logits(
            model_m=model_m,
            model_n=model_n,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        samples.append(text)

    return samples


def generate_at_multiple_alphas(
    model_m: AutoModelForCausalLM,
    model_n: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    alpha_values: List[float],
    samples_per_alpha: int = 10,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> dict:
    """
    Generate samples at multiple alpha values.

    Useful for studying how behavior changes across the interpolation range.

    Args:
        model_m: First model
        model_n: Second model
        tokenizer: Tokenizer
        prompt: Starting prompt
        alpha_values: List of alpha values to sample at
        samples_per_alpha: Number of samples per alpha
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        show_progress: Show progress bar
        device: Device to use

    Returns:
        Dictionary mapping alpha values to lists of generated texts
    """
    results = {}

    for alpha in alpha_values:
        if show_progress:
            print(f"\n--- Alpha = {alpha} ---")

        samples = generate_samples(
            model_m=model_m,
            model_n=model_n,
            tokenizer=tokenizer,
            prompt=prompt,
            alpha=alpha,
            num_samples=samples_per_alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_progress=show_progress,
            device=device,
        )
        results[alpha] = samples

    return results


def quick_compare(
    model_m: AutoModelForCausalLM,
    model_n: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    alphas: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0],
    samples_per_alpha: int = 3,
    max_new_tokens: int = 30,
    device: Optional[str] = None,
) -> None:
    """
    Quick comparison of generations at different alpha values.

    Prints samples for easy visual comparison.

    Args:
        model_m: First model
        model_n: Second model
        tokenizer: Tokenizer
        prompt: Starting prompt
        alphas: Alpha values to compare
        samples_per_alpha: Samples per alpha
        max_new_tokens: Max tokens
        device: Device to use
    """
    print(f"Prompt: {prompt!r}\n")
    print("=" * 60)

    for alpha in alphas:
        print(f"\n>>> Alpha = {alpha}")
        print("-" * 40)

        for i in range(samples_per_alpha):
            text, _ = generate_from_interpolated_logits(
                model_m=model_m,
                model_n=model_n,
                tokenizer=tokenizer,
                prompt=prompt,
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                device=device,
            )
            print(f"  [{i+1}] {text[:100]}{'...' if len(text) > 100 else ''}")

    print("\n" + "=" * 60)
