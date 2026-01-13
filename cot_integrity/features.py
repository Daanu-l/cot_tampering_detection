from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from .modeling import (
    ModelBundle,
    conditional_nll_of_prefill,
    hidden_last_token_and_nll,
    hidden_states_at_last_token,
    last_layer_hidden_states_sequence,
)


@dataclass
class PrefillFeatures:
    x: np.ndarray
    nll: float


def extract_prefill_features(
    bundle: ModelBundle, messages: list[dict[str, str]], assistant_prefill_text: str, layers: Sequence[int]
) -> PrefillFeatures:
    x, nll = hidden_last_token_and_nll(bundle, messages, assistant_prefill_text, layers)
    return PrefillFeatures(x=x, nll=nll)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def posthoc_similarity_features(clean_H: np.ndarray, clean_ids: list[int], disp_H: np.ndarray, disp_ids: list[int]) -> dict[str, float]:
    """
    Align token-id sequences and compute similarity stats between hidden states.
    """
    sm = SequenceMatcher(a=clean_ids, b=disp_ids)
    blocks = sm.get_matching_blocks()  # includes terminal (a=len, b=len, size=0)

    sims: list[float] = []
    matched = 0

    for bl in blocks:
        a0, b0, size = bl.a, bl.b, bl.size
        for k in range(size):
            i = a0 + k
            j = b0 + k
            sims.append(cosine_sim(clean_H[i], disp_H[j]))
        matched += size

    max_len = max(len(clean_ids), len(disp_ids), 1)
    frac_matched = matched / max_len
    ratio = sm.ratio()

    if sims:
        mean_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))
        p10 = float(np.percentile(sims, 10))
    else:
        mean_sim, min_sim, p10 = 0.0, 0.0, 0.0

    # A couple more robust scalars
    len_clean = float(len(clean_ids))
    len_disp = float(len(disp_ids))
    len_diff = abs(len_clean - len_disp) / max(len_clean, 1.0)

    return {
        "sim_mean": mean_sim,
        "sim_min": min_sim,
        "sim_p10": p10,
        "match_frac": float(frac_matched),
        "seq_ratio": float(ratio),
        "len_diff": float(len_diff),
    }


def extract_posthoc_features(
    bundle: ModelBundle, messages: list[dict[str, str]], clean_prefill_text: str, displayed_prefill_text: str, layer_index: int = -1
) -> dict[str, float]:
    clean_H, clean_ids = last_layer_hidden_states_sequence(bundle, messages, clean_prefill_text, layer_index=layer_index)
    disp_H, disp_ids = last_layer_hidden_states_sequence(bundle, messages, displayed_prefill_text, layer_index=layer_index)
    feats = posthoc_similarity_features(clean_H, clean_ids, disp_H, disp_ids)

    # Text-only NLL baseline on displayed trace (conditional on prompt)
    feats["nll_displayed"] = conditional_nll_of_prefill(bundle, messages, displayed_prefill_text)
    return feats


def extract_posthoc_actvec(bundle: ModelBundle, messages: list[dict[str, str]], displayed_prefill_text: str, layers: Sequence[int]) -> np.ndarray:
    """
    Posthoc activation vector: hidden states at the LAST token of the displayed trace.
    Mirrors extract_prefill_features(...) but for displayed_prefill_text.
    """
    x = hidden_states_at_last_token(bundle=bundle, messages=messages, assistant_prefill_text=displayed_prefill_text, layers=layers)
    return x
