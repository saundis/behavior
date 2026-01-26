"""
Behavior Experiments: Logit Interpolation for Model Behavior Analysis

This package provides tools for analyzing model behavior distributions
through logit interpolation and extrapolation between models.
"""

from .models import load_model, load_tokenizer, ModelPair
from .logits import get_next_token_logits, get_logits_for_sequence
from .interpolation import interpolate_logits, extrapolate_logits
from .generation import generate_from_interpolated_logits, sample_token

__all__ = [
    'load_model',
    'load_tokenizer',
    'ModelPair',
    'get_next_token_logits',
    'get_logits_for_sequence',
    'interpolate_logits',
    'extrapolate_logits',
    'generate_from_interpolated_logits',
    'sample_token',
]
