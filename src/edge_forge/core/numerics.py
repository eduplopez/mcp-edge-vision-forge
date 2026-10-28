"""Numerical comparison primitives used by export validation and drift analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPSILON = 1e-12


def as_float64(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a contiguous one-dimensional float64 view of the input."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compare empty arrays")
    return np.ascontiguousarray(array.reshape(-1))


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays, safe for zero vectors."""
    left = as_float64(reference)
    right = as_float64(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same number of elements")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _EPSILON and right_norm <= _EPSILON:
        return 1.0
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return 0.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.clip(similarity, -1.0, 1.0))


def mean_squared_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean squared error between two flattened arrays."""
    left = as_float64(reference)
    right = as_float64(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same number of elements")
    return float(np.mean(np.square(left - right)))


def mean_absolute_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean absolute error between two flattened arrays."""
    left = as_float64(reference)
    right = as_float64(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same number of elements")
    return float(np.mean(np.abs(left - right)))


def max_absolute_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """L-infinity distance between two flattened arrays."""
    left = as_float64(reference)
    right = as_float64(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same number of elements")
    return float(np.max(np.abs(left - right)))


def relative_l1_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """L1 error normalized by the reference norm."""
    left = as_float64(reference)
    right = as_float64(candidate)
    denominator = float(np.sum(np.abs(left)))
    if denominator <= _EPSILON:
        return float(np.sum(np.abs(left - right)))
    return float(np.sum(np.abs(left - right)) / denominator)


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=-1, keepdims=True)
    return np.asarray(exponentials / denominator, dtype=np.float64)


def kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Kullback-Leibler divergence D(reference || candidate) of softmax outputs."""
    left = as_float64(reference)
    right = as_float64(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same number of elements")
    probabilities = np.clip(softmax(left), _EPSILON, 1.0)
    candidate_probabilities = np.clip(softmax(right), _EPSILON, 1.0)
    divergence = np.sum(probabilities * np.log(probabilities / candidate_probabilities))
    return float(max(divergence, 0.0))


def signal_to_noise_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-noise ratio in decibels, treating the error as noise."""
    left = as_float64(reference)
    right = as_float64(candidate)
    signal = float(np.sum(np.square(left)))
    noise = float(np.sum(np.square(left - right)))
    if noise <= _EPSILON:
        return 120.0
    if signal <= _EPSILON:
        return 0.0
    return float(10.0 * np.log10(signal / noise))


def top1_agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Fraction of samples whose argmax matches over the last dimension."""
    left = np.asarray(reference)
    right = np.asarray(candidate)
    if left.shape != right.shape:
        raise ValueError("arrays must have the same shape")
    if left.ndim == 1:
        left = left.reshape(1, -1)
        right = right.reshape(1, -1)
    left_indices = np.argmax(left.reshape(left.shape[0], -1), axis=1)
    right_indices = np.argmax(right.reshape(right.shape[0], -1), axis=1)
    return float(np.mean(left_indices == right_indices))


def assert_finite(values: np.ndarray, *, label: str) -> None:
    """Raise when an array contains NaN or infinite values."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values")
