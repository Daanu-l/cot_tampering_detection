from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any


@dataclass
class TamperResult:
    tampered_text: str
    meta: dict[str, Any]


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return sents or [text.strip()]


def _severity_to_pct(severity: str) -> int:
    # Reasonable defaults for posthoc "easy/medium/hard"
    sev = (severity or "hard").lower()
    if sev == "easy":
        return 25
    if sev == "medium":
        return 50
    return 75  # hard


def tamper_token_delete(text: str, tokenizer, rng, pct: int) -> TamperResult:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n = len(ids)
    if n <= 1:
        return TamperResult(text, {"type": "token_delete", "pct": pct, "k": 0, "n_tokens": n})
    k = int(round((pct / 100.0) * n))
    if pct > 0 and n > 1:
        k = max(1, k)
    k = max(0, min(k, n - 1))
    if k == 0:
        return TamperResult(text, {"type": "token_delete", "pct": pct, "k": 0, "n_tokens": n})

    idxs = list(range(n))
    rng.shuffle(idxs)
    remove = set(idxs[:k])
    kept = [tok for i, tok in enumerate(ids) if i not in remove]
    out = tokenizer.decode(kept, skip_special_tokens=False)
    changed = out.strip() != text.strip()
    return TamperResult(out, {"type": "token_delete", "pct": pct, "k": k, "n_tokens": n, "changed": changed})


def tamper_token_insert(text: str, tokenizer, rng, pct: int, mode: str = "paper_random_vocab") -> TamperResult:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n = len(ids)
    k = int(round((pct / 100.0) * max(n, 1)))
    if pct > 0:
        k = max(1, k)
    if k <= 0:
        return TamperResult(text, {"type": "token_insert", "pct": pct, "k": 0, "n_tokens": n, "mode": mode})

    out = ids[:]
    vocab_size = getattr(tokenizer, "vocab_size", None) or 0

    if mode == "in_distribution":
        pool = ids[:] if ids else [tokenizer.eos_token_id]
        for _ in range(k):
            tok = pool[rng.randrange(len(pool))]
            pos = rng.randrange(len(out) + 1)
            out.insert(pos, tok)
    else:
        # Paper-ish: insert random vocab tokens (not sampled from the CoT)
        # Avoid very small IDs sometimes reserved for special-ish tokens.
        low = 10
        high = max(low + 1, vocab_size - 1)
        for _ in range(k):
            tok = rng.randrange(low, high) if high > low else tokenizer.eos_token_id
            pos = rng.randrange(len(out) + 1)
            out.insert(pos, tok)

    out_text = tokenizer.decode(out, skip_special_tokens=False)
    changed = out_text.strip() != text.strip()
    return TamperResult(out_text, {"type": "token_insert", "pct": pct, "k": k, "n_tokens": n, "mode": mode, "changed": changed})


def tamper_sentence_remove(text: str, rng, pct: int) -> TamperResult:
    sents = split_sentences(text)
    n = len(sents)
    if n <= 1:
        return TamperResult(text, {"type": "sentence_remove", "pct": pct, "k": 0, "n_sent": n})
    k = int(round((pct / 100.0) * n))
    # if pct>0 and there is more than 1 sentence, force at least 1 removed
    if pct > 0 and n > 1:
        k = max(1, k)
    k = max(0, min(k, n - 1))
    if k == 0:
        return TamperResult(text, {"type": "sentence_remove", "pct": pct, "k": 0, "n_sent": n})

    idxs = list(range(n))
    rng.shuffle(idxs)
    remove = set(idxs[:k])
    kept = [s for i, s in enumerate(sents) if i not in remove]
    out = " ".join(kept).strip()
    changed = out.strip() != text.strip()
    return TamperResult(out, {"type": "sentence_remove", "pct": pct, "k": k, "n_sent": n, "changed": changed})


def tamper_sentence_paraphrase_last(text: str, rng, paraphrase_fn: Callable[[str], str]) -> TamperResult:
    sents = split_sentences(text)
    if not sents:
        return TamperResult(text, {"type": "sentence_paraphrase", "ok": False})
    last = sents[-1]
    para = paraphrase_fn(last)
    if not para:
        return TamperResult(text, {"type": "sentence_paraphrase", "ok": False})
    sents[-1] = para
    out = " ".join(sents).strip()
    changed = out.strip() != text.strip()
    return TamperResult(out, {"type": "sentence_paraphrase", "ok": True, changed: changed})


def tamper_sentence_replace_pirate(text: str) -> TamperResult:
    sents = split_sentences(text)
    if not sents:
        return TamperResult("I'm a pirate!", {"type": "sentence_replace_pirate", "ok": True})
    # Replace the last sentence with something blatant.
    sents[-1] = "I'm a pirate!"
    out = " ".join(sents).strip()
    return TamperResult(out, {"type": "sentence_replace_pirate", "ok": True})


def tamper_dispatch(
    tamper_type: str,
    text: str,
    tokenizer,
    rng,
    pct: int | None = None,
    severity: str | None = None,
    paraphrase_fn: Callable[[str], str] | None = None,
    token_insert_mode: str = "paper_random_vocab",
) -> TamperResult:
    """
    pct: explicit percent (prefill sweep)
    severity: fallback for posthoc script ("easy/medium/hard")
    """
    if pct is None and severity is not None:
        pct = _severity_to_pct(severity)

    if tamper_type == "token_delete":
        assert pct is not None, "token_delete requires pct or severity"
        return tamper_token_delete(text, tokenizer, rng, pct)

    if tamper_type == "token_insert":
        assert pct is not None, "token_insert requires pct or severity"
        return tamper_token_insert(text, tokenizer, rng, pct, mode=token_insert_mode)

    if tamper_type == "sentence_remove":
        assert pct is not None, "sentence_remove requires pct or severity"
        return tamper_sentence_remove(text, rng, pct)

    if tamper_type == "sentence_paraphrase":
        if paraphrase_fn is None:
            raise ValueError("sentence_paraphrase requires paraphrase_fn")
        return tamper_sentence_paraphrase_last(text, rng, paraphrase_fn)

    if tamper_type == "sentence_replace_pirate":
        return tamper_sentence_replace_pirate(text)

    raise ValueError(f"Unknown tamper_type: {tamper_type}")
