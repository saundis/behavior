"""
Core logit interpolation and extrapolation functions.

This module implements the key technique: combining logit distributions
from two models to amplify behavioral differences.

The core formula:
    L_alpha = alpha * L_M + (1 - alpha) * L_N

Where:
    - L_M: logits from model M
    - L_N: logits from model N
    - alpha in [0, 1]: interpolation between models
    - alpha > 1 or alpha < 0: extrapolation (amplifies differences)
"""

import torch
from typing import Optional, Tuple
import torch.nn.functional as F


def interpolate_logits(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Interpolate or extrapolate between two logit distributions.

    The formula: L_alpha = alpha * L_M + (1 - alpha) * L_N

    Args:
        logits_m: Logits from model M, shape (vocab_size,) or (batch, vocab_size)
        logits_n: Logits from model N, same shape as logits_m
        alpha: Interpolation parameter
            - alpha = 0: returns L_N (model N's distribution)
            - alpha = 1: returns L_M (model M's distribution)
            - alpha in (0, 1): interpolates between models
            - alpha > 1: extrapolates toward M (amplifies M's tendencies)
            - alpha < 0: extrapolates toward N (amplifies N's tendencies)

    Returns:
        Interpolated logits with same shape as input
    """
    return alpha * logits_m + (1 - alpha) * logits_n


def interpolate_probs(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Interpolate in probability space rather than logit space.

    First converts logits to probabilities, then interpolates.
    This is an alternative approach that may have different properties.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        alpha: Interpolation parameter
        temperature: Temperature for softmax

    Returns:
        Interpolated probability distribution
    """
    probs_m = F.softmax(logits_m / temperature, dim=-1)
    probs_n = F.softmax(logits_n / temperature, dim=-1)

    # Interpolate probabilities
    interpolated_probs = alpha * probs_m + (1 - alpha) * probs_n

    # Ensure valid probability distribution (may be needed for extrapolation)
    interpolated_probs = torch.clamp(interpolated_probs, min=1e-10)
    interpolated_probs = interpolated_probs / interpolated_probs.sum(dim=-1, keepdim=True)

    return interpolated_probs


def extrapolate_logits(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    strength: float,
    direction: str = "m",
) -> torch.Tensor:
    """
    Convenience function for extrapolation with clearer semantics.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        strength: How much to extrapolate (1.0 = normal, 2.0 = double the difference)
        direction: 'm' to extrapolate toward M, 'n' to extrapolate toward N

    Returns:
        Extrapolated logits

    Example:
        If M is slightly more likely to refuse than N, extrapolating toward M
        with strength=2.0 should make refusal even more likely.
    """
    if direction.lower() == "m":
        # Extrapolate toward M: alpha > 1
        alpha = strength
    elif direction.lower() == "n":
        # Extrapolate toward N: alpha < 0
        alpha = 1 - strength
    else:
        raise ValueError(f"Direction must be 'm' or 'n', got {direction}")

    return interpolate_logits(logits_m, logits_n, alpha)


def compute_interpolation_range(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    alpha_values: list,
    return_probs: bool = True,
) -> dict:
    """
    Compute interpolated distributions for a range of alpha values.

    Useful for analyzing how behavior changes across the interpolation range.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        alpha_values: List of alpha values to compute
        return_probs: If True, return probabilities; otherwise return logits

    Returns:
        Dictionary mapping alpha values to distributions
    """
    results = {}
    for alpha in alpha_values:
        interp_logits = interpolate_logits(logits_m, logits_n, alpha)
        if return_probs:
            results[alpha] = F.softmax(interp_logits, dim=-1)
        else:
            results[alpha] = interp_logits
    return results


def find_token_probability_at_alpha(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    token_id: int,
    alpha: float,
) -> float:
    """
    Find the probability of a specific token at a given alpha.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        token_id: The token ID to check
        alpha: Interpolation parameter

    Returns:
        Probability of the token
    """
    interp_logits = interpolate_logits(logits_m, logits_n, alpha)
    probs = F.softmax(interp_logits, dim=-1)
    return probs[token_id].item()


def analyze_token_across_alpha(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    token_id: int,
    alpha_range: Tuple[float, float] = (-2.0, 3.0),
    num_points: int = 50,
) -> dict:
    """
    Analyze how a token's probability changes across alpha values.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        token_id: The token ID to analyze
        alpha_range: (min_alpha, max_alpha) range to analyze
        num_points: Number of points to sample

    Returns:
        Dictionary with alpha values and corresponding probabilities
    """
    alphas = torch.linspace(alpha_range[0], alpha_range[1], num_points)
    probs = []

    for alpha in alphas:
        prob = find_token_probability_at_alpha(logits_m, logits_n, token_id, alpha.item())
        probs.append(prob)

    return {
        "alphas": alphas.tolist(),
        "probabilities": probs,
        "token_id": token_id,
    }


def find_alpha_for_target_probability(
    logits_m: torch.Tensor,
    logits_n: torch.Tensor,
    token_id: int,
    target_prob: float,
    search_range: Tuple[float, float] = (-5.0, 5.0),
    tolerance: float = 0.001,
    max_iterations: int = 100,
) -> Optional[float]:
    """
    Find the alpha value that gives a target probability for a token.

    Uses binary search to find the alpha value.

    Args:
        logits_m: Logits from model M
        logits_n: Logits from model N
        token_id: The token ID
        target_prob: Desired probability
        search_range: Range to search
        tolerance: Acceptable error
        max_iterations: Maximum search iterations

    Returns:
        Alpha value that achieves target probability, or None if not found
    """
    low, high = search_range

    # Check if target is achievable
    prob_low = find_token_probability_at_alpha(logits_m, logits_n, token_id, low)
    prob_high = find_token_probability_at_alpha(logits_m, logits_n, token_id, high)

    if not (min(prob_low, prob_high) <= target_prob <= max(prob_low, prob_high)):
        return None  # Target not achievable in this range

    # Binary search
    for _ in range(max_iterations):
        mid = (low + high) / 2
        prob_mid = find_token_probability_at_alpha(logits_m, logits_n, token_id, mid)

        if abs(prob_mid - target_prob) < tolerance:
            return mid

        # Determine search direction based on monotonicity
        if (prob_high > prob_low and prob_mid < target_prob) or \
           (prob_high < prob_low and prob_mid > target_prob):
            low = mid
        else:
            high = mid

    return (low + high) / 2  # Return best approximation


class LogitInterpolator:
    """
    Class for managing interpolation experiments between two models.

    Caches logits and provides convenient methods for analysis.
    """

    def __init__(self, logits_m: torch.Tensor, logits_n: torch.Tensor, tokenizer=None):
        """
        Initialize with logits from two models.

        Args:
            logits_m: Logits from model M
            logits_n: Logits from model N
            tokenizer: Optional tokenizer for decoding
        """
        self.logits_m = logits_m
        self.logits_n = logits_n
        self.tokenizer = tokenizer

    def interpolate(self, alpha: float) -> torch.Tensor:
        """Get interpolated logits at given alpha."""
        return interpolate_logits(self.logits_m, self.logits_n, alpha)

    def get_probs(self, alpha: float) -> torch.Tensor:
        """Get probability distribution at given alpha."""
        return F.softmax(self.interpolate(alpha), dim=-1)

    def get_top_k(self, alpha: float, k: int = 10) -> list:
        """Get top-k tokens and probabilities at given alpha."""
        probs = self.get_probs(alpha)
        top_probs, top_indices = torch.topk(probs, k)

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = self.tokenizer.decode([idx]) if self.tokenizer else f"[{idx}]"
            results.append({"token": token_str, "token_id": idx, "probability": prob})

        return results

    def compare_at_alphas(self, alphas: list, k: int = 5) -> dict:
        """Compare top-k tokens across multiple alpha values."""
        return {alpha: self.get_top_k(alpha, k) for alpha in alphas}
