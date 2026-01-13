from __future__ import annotations

import re

# Remove common control tokens emitted by Qwen/DeepSeek tokenizers
# Example: <｜end▁of▁sentence｜> (fullwidth vertical bar U+FF5C)
_QWEN_CTRL_RE = re.compile(r"<\uFF5C[^>]*\uFF5C>")
# Example: <|eot_id|>, <|endoftext|>, etc
_HF_CTRL_RE = re.compile(r"<\|[^>]*\|>")

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_MOD_RE = re.compile(r"<modified>\s*(yes|no)\s*</modified>", re.IGNORECASE)
_BOXED_START_RE = re.compile(r"\\boxed\s*{", re.IGNORECASE)

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")


def strip_chat_control_tokens(text: str) -> str:
    """
    Remove tokenizer control tokens that interfere with parsing.
    Does NOT remove your normal tags like <think> / <modified>.
    """
    if not text:
        return text
    text = _HF_CTRL_RE.sub("", text)
    text = _QWEN_CTRL_RE.sub("", text)
    return text


def extract_modified_flag(text: str) -> bool | None:
    if not text:
        return None
    t = strip_chat_control_tokens(text)
    matches = _MOD_RE.findall(t)
    if not matches:
        return None
    last = matches[-1].strip().lower()
    if last == "yes":
        return True
    if last == "no":
        return False
    return None


def extract_think(text: str) -> str | None:
    if not text:
        return None
    t = strip_chat_control_tokens(text)
    m = _THINK_RE.search(t)
    if m:
        out = m.group(1).strip()
        return out or None

    # Fallback: if model never produced </think>, take everything after <think> up to \boxed{...} if present
    lo = t.lower()
    start = lo.find("<think>")
    if start == -1:
        return None

    after = t[start + len("<think>") :]
    m2 = re.search(r"\\boxed\s*{", after)
    if m2:
        after = after[: m2.start()]
    out = after.strip()
    return out or None


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the LAST \\boxed{...}, supports nested braces.
    Also supports '\\boxed { ... }' (space between).
    """
    if not text:
        return None
    t = strip_chat_control_tokens(text)

    # Find last occurrence of \boxed{ with optional whitespace
    last_idx = -1
    for m in re.finditer(r"\\boxed\s*{", t, flags=re.IGNORECASE):
        last_idx = m.start()

    if last_idx == -1:
        return None

    # Move to the '{'
    brace_open = t.find("{", last_idx)
    if brace_open == -1:
        return None

    depth = 1
    out: list[str] = []
    i = brace_open + 1
    while i < len(t) and depth > 0:
        ch = t[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1

    ans = "".join(out).strip()
    return ans or None


def extract_choice_fallback(text: str) -> str | None:
    if not text:
        return None
    t = strip_chat_control_tokens(text).upper()
    m = re.findall(r"\b([ABCD])\b", t)
    return m[-1] if m else None


def normalize_mmlu_choice(s: str) -> str | None:
    if not s:
        return None
    s = strip_chat_control_tokens(s).strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    boxed = extract_boxed_answer(s)
    if boxed:
        boxed = boxed.strip().upper()
        if boxed in {"A", "B", "C", "D"}:
            return boxed
    m = re.search(r"\b([ABCD])\b", s)
    return m.group(1) if m else None


def simple_math_normalize(s: str) -> str:
    s = strip_chat_control_tokens(s or "")
    s = s.strip().replace(" ", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("−", "-")
    return s


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    t = strip_chat_control_tokens(text).strip()
    parts = _SENT_SPLIT_RE.split(t)
    return [p.strip() for p in parts if p.strip()]


def last_sentence(text: str) -> str | None:
    sents = split_sentences(text)
    return sents[-1] if sents else None
