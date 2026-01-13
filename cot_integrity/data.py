from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from datasets import load_dataset

from .utils import stable_hash


@dataclass(frozen=True)
class Example:
    uid: str
    dataset: str  # "math500" | "mmlu"
    subject: str | None
    prompt: str  # fully formatted problem text
    choices: list[str] | None = None
    correct_index: int | None = None
    ground_truth: str | None = None


DEFAULT_MMLU_SUBJECTS: list[str] = [
    # include marketing because Cywiński used it in sandbagging section
    "marketing",
    "abstract_algebra",
    "college_medicine",
    "computer_security",
    "high_school_mathematics",
    "logical_fallacies",
]


def load_math500(n: int, seed: int = 0, dataset_name: str = "PrimeIntellect/MATH-500", split: str = "train") -> list[Example]:
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)

    out: list[Example] = []
    for row in ds.select(range(min(n, len(ds)))):
        prompt = row.get("prompt") or row.get("problem") or row.get("question")
        if prompt is None:
            continue

        gt = None
        vi = row.get("verification_info")
        if isinstance(vi, str):
            try:
                vi_json = json.loads(vi)
                gt = vi_json.get("ground_truth")
            except Exception:
                gt = None
        elif isinstance(vi, dict):
            gt = vi.get("ground_truth")

        uid = row.get("problem_id") or stable_hash(prompt)

        out.append(
            Example(uid=f"math500:{uid}", dataset="math500", subject=None, prompt=str(prompt), choices=None, correct_index=None, ground_truth=gt)
        )
    return out


def _try_load_mmlu_split(subject: str, split_candidates: Sequence[str]) -> Any:
    last_err = None
    for sp in split_candidates:
        try:
            return load_dataset("cais/mmlu", subject, split=sp), sp
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load MMLU subject={subject} with splits={split_candidates}") from last_err


def load_mmlu(
    n_total: int, seed: int = 0, subjects: list[str] | None = None, split_candidates: Sequence[str] = ("test", "validation", "dev")
) -> list[Example]:
    subjects = subjects or DEFAULT_MMLU_SUBJECTS
    out: list[Example] = []

    # 1) First balanced pass
    per_subject = max(1, n_total // max(1, len(subjects)))

    subject_datasets = []
    for subj in subjects:
        ds, used_split = _try_load_mmlu_split(subj, split_candidates)
        ds = ds.shuffle(seed=seed)
        subject_datasets.append((subj, used_split, ds))

        take_n = min(per_subject, n_total - len(out), len(ds))
        for row in ds.select(range(take_n)):
            q = row["question"]
            choices = row["choices"]
            ans = int(row["answer"])
            prompt = format_mmlu_prompt(q, choices)
            uid = stable_hash(f"mmlu:{subj}:{used_split}:{q}")  # NOTE: prefixed uid

            out.append(
                Example(
                    uid=uid,
                    dataset="mmlu",
                    subject=subj,
                    prompt=prompt,
                    choices=list(choices),
                    correct_index=ans,
                    ground_truth=["A", "B", "C", "D"][ans],
                )
            )
            if len(out) >= n_total:
                return out[:n_total]

    # 2) Top-up pass: fill remaining from subjects round-robin
    if len(out) < n_total:
        used_uids = set(e.uid for e in out)
        # Keep iterating subjects and taking next examples until filled
        for subj, used_split, ds in subject_datasets:
            if len(out) >= n_total:
                break
            # Start after the per_subject slice
            start = per_subject
            remaining = min(len(ds) - start, n_total - len(out))
            if remaining <= 0:
                continue

            for row in ds.select(range(start, start + remaining)):
                q = row["question"]
                choices = row["choices"]
                ans = int(row["answer"])
                prompt = format_mmlu_prompt(q, choices)
                uid = stable_hash(f"mmlu:{subj}:{used_split}:{q}")

                if uid in used_uids:
                    continue
                used_uids.add(uid)

                out.append(
                    Example(
                        uid=uid,
                        dataset="mmlu",
                        subject=subj,
                        prompt=prompt,
                        choices=list(choices),
                        correct_index=ans,
                        ground_truth=["A", "B", "C", "D"][ans],
                    )
                )
                if len(out) >= n_total:
                    break

    return out[:n_total]


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [f"Question: {question}", ""]
    for i, c in enumerate(choices[:4]):
        lines.append(f"{letters[i]}. {c}")
    lines.append("")
    lines.append("Choose the best answer. Put the letter in \\boxed{...}.")
    return "\n".join(lines)
