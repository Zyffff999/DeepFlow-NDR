from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.pipeline.extractor import ExtractedFlow


@dataclass
class TensorizedFlow:
    byte_matrix: np.ndarray
    iat_series: np.ndarray
    size_series: np.ndarray
    global_stats: np.ndarray
    global_stats_keys: List[str]


def _pad_series(values: Sequence[float], length: int) -> np.ndarray:
    padded = np.zeros(length, dtype=np.float32)
    if not values:
        return padded
    trimmed = np.asarray(values[:length], dtype=np.float32)
    padded[: trimmed.size] = trimmed
    return padded


def tensorize_flow(
    flow: ExtractedFlow,
    seq_len: int,
    global_stats_keys: Sequence[str] | None = None,
    normalize_bytes: bool = True,
) -> TensorizedFlow:
    if global_stats_keys:
        keys = list(global_stats_keys)
    else:
        keys = sorted(flow.global_stats.keys())
    global_stats_vector = np.asarray(
        [float(flow.global_stats.get(key, 0.0)) for key in keys], dtype=np.float32
    )
    byte_matrix = flow.byte_matrix.astype(np.float32)
    if normalize_bytes:
        byte_matrix = byte_matrix / 255.0
    return TensorizedFlow(
        byte_matrix=byte_matrix,
        iat_series=_pad_series(flow.iat_series, seq_len),
        size_series=_pad_series(flow.size_series, seq_len),
        global_stats=global_stats_vector,
        global_stats_keys=keys,
    )


__all__ = ["TensorizedFlow", "tensorize_flow"]
