"""
Run the *prefill/cache-mismatch* CoT-tampering experiment.

This script:
1) Generates clean CoTs for problems.
2) Splits the CoT; tampers the first half (token/sentence edits).
3) Continues generation under a prefill mismatch threat model.
4) Extracts features and trains linear probes (LR) to detect tampering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from cot_integrity.data import load_math500, load_mmlu
from cot_integrity.detectors import fit_logreg_classifier, eval_classifier
from cot_integrity.features import extract_prefill_features
from cot_integrity.modeling import generate_assistant_text, load_model
from cot_integrity.parsing import (
    extract_boxed_answer,
    extract_modified_flag,
    extract_think,
    normalize_mmlu_choice,
    simple_math_normalize,
)
from cot_integrity.prompts import (
    messages_base_solve,
    messages_cywinski_awareness,
    messages_cywinski_sentence_replacement,
    messages_llm_judge,
    messages_paraphrase_sentence,
)
from cot_integrity.tamper import tamper_dispatch
from cot_integrity.utils import ensure_dir, read_jsonl, set_seed, stable_hash, write_jsonl


def split_cot_in_half(cot_text: str, tokenizer) -> str:
    """
    Split near half the tokens, but snap to the nearest sentence boundary
    so sentence_remove/paraphrase operate on real sentences.
    """
    ids = tokenizer(cot_text, add_special_tokens=False).input_ids
    n = len(ids)
    if n <= 16:
        return cot_text.strip()

    target = n // 2
    prefix = tokenizer.decode(ids[:target], skip_special_tokens=False)

    m = list(re.finditer(r"(?<=[\.\!\?])\s+", prefix))
    if m:
        prefix = prefix[: m[-1].end()]

    prefix = prefix.strip()
    if len(prefix) < 40:
        prefix = tokenizer.decode(ids[: max(8, target)], skip_special_tokens=False).strip()
    return prefix


def build_prefill_text(cot_half: str) -> str:
    # leave <think> open for continuation (paper-style)
    return "<think>\n" + cot_half.strip() + "\n"


def make_paraphrase_fn(bundle, paraphrase_temperature: float = 0.7, max_new_tokens: int = 96):
    def paraphrase(sentence: str) -> str:
        # Stronger instruction for "single sentence, no fluff"
        msgs = messages_paraphrase_sentence(sentence)
        assistant, _full = generate_assistant_text(
            bundle,
            msgs,
            assistant_prefill_text=None,
            max_new_tokens=max_new_tokens,
            temperature=paraphrase_temperature,
            top_p=0.95,
            skip_special_tokens=False,
        )
        # take first non-empty line
        lines = [ln.strip() for ln in assistant.split("\n") if ln.strip()]
        out = lines[0] if lines else assistant.strip()
        out = out.strip().strip('"')
        return out

    return paraphrase


def run_self_report_rollout(
    bundle, problem_prompt: str, assistant_prefill: str, max_new_tokens: int, temperature: float, top_p: float, gen_seed: int | None = None
):
    msgs = messages_cywinski_awareness(problem_prompt)

    assistant_text, full_text = generate_assistant_text(
        bundle,
        msgs,
        assistant_prefill_text=assistant_prefill,
        max_new_tokens=max_new_tokens,
        temperature=temperature,  # <-- sampled, not greedy
        top_p=top_p,
        skip_special_tokens=False,  # keep raw; we’ll parse robustly
        gen_seed=gen_seed,
        stop_strings=["</modified>"],  # stop right after the closing tag
    )

    gen_text = assistant_prefill + (assistant_text or "")

    flag = extract_modified_flag(gen_text) or extract_modified_flag(full_text)
    parse_ok = 1 if flag is not None else 0

    boxed = extract_boxed_answer(gen_text) or extract_boxed_answer(full_text)

    return flag, parse_ok, boxed, assistant_text, full_text


def maybe_llm_judge(bundle, problem_prompt: str, reasoning_trace: str, max_new_tokens: int = 8) -> bool | None:
    msgs = messages_llm_judge(problem_prompt, reasoning_trace)
    out, _ = generate_assistant_text(
        bundle, msgs, assistant_prefill_text=None, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, skip_special_tokens=True
    )
    tok = out.strip().split()[0].upper() if out.strip() else ""
    if tok == "YES":
        return True
    if tok == "NO":
        return False
    return None


def answer_correct(dataset: str, gt: str | None, boxed: str | None) -> int | None:
    """
    Very lightweight correctness. If you already have a better checker, use it.

    - For MMLU: compare A/B/C/D
    - For math: normalized string match (not symbolic equivalence)
    """
    if gt is None or boxed is None:
        return None

    if dataset == "mmlu":
        gt_norm = normalize_mmlu_choice(gt)
        pred_norm = normalize_mmlu_choice(boxed)
        if gt_norm is None or pred_norm is None:
            return None
        return int(gt_norm == pred_norm)

    # math-ish
    return int(simple_math_normalize(gt) == simple_math_normalize(boxed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_math", type=int, default=60)
    ap.add_argument("--n_mmlu", type=int, default=60)
    ap.add_argument("--mmlu_subjects", type=str, nargs="*", default=None)

    ap.add_argument(
        "--tamper_types",
        type=str,
        nargs="+",
        default=["token_delete", "token_insert", "sentence_remove", "sentence_paraphrase"],  # , "sentence_replace_pirate"]
    )

    # Paper-style sweep (0..90)
    ap.add_argument(
        "--tamper_pcts", type=int, nargs="+", default=[20], help="Percent modification sweep used for token_delete/token_insert/sentence_remove."
    )

    ap.add_argument("--layers", type=int, nargs="+", default=[-1, -4, -8, -16, -32])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    ap.add_argument("--skip_self_report", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")

    ap.add_argument("--max_new_tokens_base", type=int, default=1024)
    ap.add_argument("--max_new_tokens_self", type=int, default=2048)
    ap.add_argument("--recompute_rollouts", action="store_true", help="Ignore rollouts_prefill.jsonl and recompute self-report rollouts.")
    ap.add_argument("--self_report_temperature", type=float, default=0.7)
    ap.add_argument("--self_report_top_p", type=float, default=0.95)
    ap.add_argument("--token_insert_mode", type=str, default="in_distribution", choices=["paper_random_vocab", "in_distribution"])
    ap.add_argument("--model_preset", type=str, default="auto", choices=["auto", "deepseek", "qwen_qwq", "nemotron", "openthinker"])
    ap.add_argument(
        "--self_report_samples",
        type=int,
        default=1,
        help="How many sampled self-report rollouts per trial (1 matches LW most closely; 3 helps you see YES sooner).",
    )
    ap.add_argument("--save_examples", type=int, default=100, help="How many illustrative examples to save for the report.")
    ap.add_argument("--save_yes_examples", type=int, default=60, help="How many self-report YES examples to prioritize (if available).")

    args = ap.parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    bundle = load_model(args.model, device_map=args.device_map, torch_dtype=args.torch_dtype)
    model_lower = args.model.lower()
    if args.model_preset == "auto":
        if "deepseek" in model_lower:
            args.model_preset = "deepseek"
        elif "qwq" in model_lower or "qwen" in model_lower:
            args.model_preset = "qwen_qwq"
        elif "nemotron" in model_lower:
            args.model_preset = "nemotron"
        elif "openthinker" in model_lower or "open-thoughts" in model_lower:
            args.model_preset = "openthinker"
        else:
            args.model_preset = "qwen_qwq"

    # Good defaults:
    # - DeepSeek/QwQ/OpenThinker typically like temp 0.6 top_p 0.95 for reasoning
    # - Nemotron model card explicitly recommends temp=0.6 top_p=0.95 for reasoning-on mode
    #   (we use same for self-report generation too)
    if args.model_preset in ["deepseek", "qwen_qwq", "openthinker", "nemotron"]:
        # keep your base solve generation as-is (already temp 0.6 / top_p 0.95)
        pass

    # Load data (IMPORTANT: to compute answer correctness, your Example objects should have ex.answer)
    examples = []
    if args.n_math > 0:
        examples += load_math500(args.n_math, seed=args.seed)
    if args.n_mmlu > 0:
        examples += load_mmlu(args.n_mmlu, seed=args.seed, subjects=args.mmlu_subjects)

    print(f"Requested n_math={args.n_math} n_mmlu={args.n_mmlu}")
    print(
        f"Loaded total examples={len(examples)} "
        f"(math={sum(e.dataset == 'math500' for e in examples)}, "
        f"mmlu={sum(e.dataset == 'mmlu' for e in examples)})"
    )

    # Cache baseline generations
    base_path = out_dir / "baseline_generations.jsonl"
    baseline = {r["uid"]: r for r in read_jsonl(base_path)}

    paraphrase_fn = make_paraphrase_fn(bundle) if ("sentence_paraphrase" in args.tamper_types) else None

    new_rows = []
    for ex in tqdm(examples, desc="Generating baselines"):
        if ex.uid in baseline:
            continue

        msgs = messages_base_solve(ex.prompt)
        think_prefill = "<think>\n"

        assistant_text, _full = generate_assistant_text(
            bundle,
            msgs,
            assistant_prefill_text=think_prefill,
            max_new_tokens=args.max_new_tokens_base,
            temperature=0.6,
            top_p=0.95,
            skip_special_tokens=False,
        )

        cot = extract_think(think_prefill + assistant_text)
        if cot is None or len(cot.strip()) < 5:
            continue

        cot_half = split_cot_in_half(cot, bundle.tokenizer)

        row = {
            "uid": ex.uid,
            "dataset": ex.dataset,
            "subject": ex.subject,
            "prompt": ex.prompt,
            "answer": getattr(ex, "ground_truth", None),  # requires loader support
            "cot_full": cot,
            "cot_half": cot_half,
        }
        new_rows.append(row)

    if new_rows:
        all_rows = list(baseline.values()) + new_rows
        write_jsonl(base_path, all_rows)
        baseline = {r["uid"]: r for r in all_rows}

    # Build trials (paper-style: sweep % for token_delete/token_insert/sentence_remove)
    trials_path = out_dir / "trials_prefill.jsonl"
    trials: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    sweep_types = {"token_delete", "token_insert", "sentence_remove"}

    for uid, b in baseline.items():
        cot_half = b["cot_half"]
        for tt in args.tamper_types:
            if tt in sweep_types:
                for pct in args.tamper_pcts:
                    # control
                    trials.append(
                        dict(
                            trial_id=stable_hash(f"{uid}:{tt}:{pct}:control"),
                            base_uid=uid,
                            dataset=b["dataset"],
                            subject=b.get("subject"),
                            prompt=b["prompt"],
                            answer=b.get("answer"),
                            tamper_type=tt,
                            tamper_pct=int(pct),
                            is_tampered=0,
                            cot_half=cot_half,
                            tampered_half=cot_half,
                            tamper_meta={"type": tt, "pct": int(pct), "control": True},
                        )
                    )

                    # tampered (skip pct=0)
                    if pct > 0:
                        tam = tamper_dispatch(
                            tamper_type=tt,
                            text=cot_half,
                            tokenizer=bundle.tokenizer,
                            rng=rng,
                            pct=int(pct),
                            paraphrase_fn=None,
                            token_insert_mode=args.token_insert_mode,
                        )
                        trials.append(
                            dict(
                                trial_id=stable_hash(f"{uid}:{tt}:{pct}:tampered"),
                                base_uid=uid,
                                dataset=b["dataset"],
                                subject=b.get("subject"),
                                prompt=b["prompt"],
                                answer=b.get("answer"),
                                tamper_type=tt,
                                tamper_pct=int(pct),
                                is_tampered=1,
                                cot_half=cot_half,
                                tampered_half=tam.tampered_text,
                                tamper_meta=tam.meta,
                            )
                        )
            else:
                # sentence_paraphrase: one fixed-strength corruption (paper replaces last sentence)
                trials.append(
                    dict(
                        trial_id=stable_hash(f"{uid}:{tt}:control"),
                        base_uid=uid,
                        dataset=b["dataset"],
                        subject=b.get("subject"),
                        prompt=b["prompt"],
                        answer=b.get("answer"),
                        tamper_type=tt,
                        tamper_pct=None,
                        is_tampered=0,
                        cot_half=cot_half,
                        tampered_half=cot_half,
                        tamper_meta={"type": tt, "control": True},
                    )
                )
                tam = tamper_dispatch(tamper_type=tt, text=cot_half, tokenizer=bundle.tokenizer, rng=rng, pct=None, paraphrase_fn=paraphrase_fn)
                trials.append(
                    dict(
                        trial_id=stable_hash(f"{uid}:{tt}:tampered"),
                        base_uid=uid,
                        dataset=b["dataset"],
                        subject=b.get("subject"),
                        prompt=b["prompt"],
                        answer=b.get("answer"),
                        tamper_type=tt,
                        tamper_pct=None,
                        is_tampered=1,
                        cot_half=cot_half,
                        tampered_half=tam.tampered_text,
                        tamper_meta=tam.meta,
                    )
                )

    write_jsonl(trials_path, trials)

    # Group split by base_uid
    groups = np.array([t["base_uid"] for t in trials])
    y = np.array([t["is_tampered"] for t in trials], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(trials)), y, groups=groups))

    # Rollout cache to avoid re-running self-report when re-training probes
    rollout_cache_path = out_dir / "rollouts_prefill.jsonl"
    rollout_cache = {r["trial_id"]: r for r in read_jsonl(rollout_cache_path)}

    rows = []
    X_act = []
    y_vec = []

    failures_path = out_dir / "self_report_failures.jsonl"
    print(f"Built {len(trials)} trials from {len(baseline)} baselines.")
    if len(trials) > 20000:
        raise RuntimeError("Too many trials. Did you forget to pass --tamper_pcts 20 ?")

    for t in tqdm(trials, desc="Extracting features (prefill)"):
        if t["tamper_type"] in {"sentence_replace_pirate", "sentence_paraphrase"}:
            msgs = messages_cywinski_sentence_replacement(t["prompt"])
        else:
            msgs = messages_cywinski_awareness(t["prompt"])

        assistant_prefill = build_prefill_text(t["tampered_half"])

        feats = extract_prefill_features(bundle, msgs, assistant_prefill, layers=args.layers)

        row = {
            "trial_id": t["trial_id"],
            "base_uid": t["base_uid"],
            "dataset": t["dataset"],
            "subject": t.get("subject"),
            "tamper_type": t["tamper_type"],
            "tamper_pct": t.get("tamper_pct"),
            "is_tampered": int(t["is_tampered"]),
            "nll_prefill": feats.nll,
        }

        # Behavioral rollout (answer + self-report) — cached
        # Behavioral rollout (answer + self-report) — cached
        if not args.skip_self_report:
            use_cache = (t["trial_id"] in rollout_cache) and (not args.recompute_rollouts)

            if use_cache:
                rc = rollout_cache[t["trial_id"]]
                flag_int = rc.get("self_report_flag", None)
                parse_ok = rc.get("self_report_parse_ok", 0)
                boxed = rc.get("boxed_answer", None)
                assistant_text = rc.get("assistant_text", "")
                full_text = rc.get("full_text", "")
            else:
                seed_i = int(stable_hash(t["trial_id"]), 16) % (2**32)
                flag_bool, parse_ok, boxed, assistant_text, full_text = run_self_report_rollout(
                    bundle,
                    t["prompt"],
                    assistant_prefill,
                    max_new_tokens=args.max_new_tokens_self,
                    temperature=args.self_report_temperature,
                    top_p=args.self_report_top_p,
                    gen_seed=seed_i,
                )
                flag_int = None if flag_bool is None else int(flag_bool)

                rollout_cache[t["trial_id"]] = {
                    "trial_id": t["trial_id"],
                    "self_report_flag": (0 if flag_int is None else int(flag_int)),
                    "self_report_parse_ok": int(parse_ok),
                    "boxed_answer": boxed,
                    "assistant_text": assistant_text,  # keep full text for debugging
                    "full_text": full_text,
                }

                if parse_ok == 0:
                    with Path(failures_path).open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "trial_id": t["trial_id"],
                                    "tamper_type": t["tamper_type"],
                                    "tamper_pct": t.get("tamper_pct"),
                                    "is_tampered": int(t["is_tampered"]),
                                    "assistant_prefill_tail": assistant_prefill[-2000:],
                                    "assistant_text_tail": (assistant_text or "")[-8000:],
                                    "full_text_tail": (full_text or "")[-8000:],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            gt = t.get("answer")
            row["answer_correct"] = answer_correct(t["dataset"], gt, boxed)
            row["self_report_parse_ok"] = int(parse_ok)
            row["self_report_modified"] = 0 if flag_int is None else int(flag_int)
            row["boxed_answer"] = boxed

        if not args.skip_judge:
            judge = maybe_llm_judge(bundle, t["prompt"], t["tampered_half"])
            row["llm_judge_modified"] = None if judge is None else int(judge)

        rows.append(row)
        X_act.append(feats.x)
        y_vec.append(int(t["is_tampered"]))

    # Save rollout cache
    write_jsonl(rollout_cache_path, list(rollout_cache.values()))

    # Save activations for reuse
    X_act_np = np.stack(X_act, axis=0).astype(np.float16)
    np.save(out_dir / "X_act_prefill_fp16.npy", X_act_np)
    np.save(out_dir / "y_prefill.npy", np.array(y_vec, dtype=np.int64))
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    # ---- Save layer slice metadata for offline layer sweeps ----
    # X_act_np shape: [n_trials, D] where D = len(layers) * hidden_size
    D = int(X_act_np.shape[1])
    L = len(args.layers)
    assert D % L == 0, f"Feature dim {D} not divisible by n_layers {L}"
    d_per = D // L

    layer_slices = {}
    for i, li in enumerate(args.layers):
        layer_slices[str(li)] = {"start": int(i * d_per), "end": int((i + 1) * d_per), "dim": int(d_per)}

    (out_dir / "layer_slices.json").write_text(
        json.dumps({"layers": args.layers, "d_per_layer": d_per, "slices": layer_slices}, indent=2), encoding="utf-8"
    )

    X_act = np.stack(X_act, axis=0)
    y_vec = np.array(y_vec, dtype=int)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "features_prefill.csv", index=False)
    (out_dir / "trial_id_order.txt").write_text("\n".join(df["trial_id"].tolist()), encoding="utf-8")

    # ---- Save illustrative examples for the report ----
    trial_by_id = {t["trial_id"]: t for t in trials}
    roll_by_id = {r["trial_id"]: r for r in rollout_cache.values()}

    examples = []

    # 1) prioritize YES self-reports
    if "self_report_modified" in df.columns:
        yes_df = df[df["self_report_modified"] == 1].copy()
        for _, r in yes_df.head(args.save_yes_examples).iterrows():
            tid = r["trial_id"]
            tr = trial_by_id.get(tid, {})
            ro = roll_by_id.get(tid, {})
            examples.append({
                "trial_id": tid,
                "tamper_type": tr.get("tamper_type"),
                "tamper_pct": tr.get("tamper_pct"),
                "is_tampered": tr.get("is_tampered"),
                "dataset": tr.get("dataset"),
                "prompt": tr.get("prompt"),
                "cot_half": tr.get("cot_half"),
                "tampered_half": tr.get("tampered_half"),
                "self_report_flag": ro.get("self_report_flag"),
                "self_report_parse_ok": ro.get("self_report_parse_ok"),
                "boxed_answer": ro.get("boxed_answer"),
                "assistant_text": ro.get("assistant_text"),
                "full_text": ro.get("full_text"),
            })

    # 2) fill remaining with diverse random examples (balanced across tamper types)
    need = max(0, args.save_examples - len(examples))
    if need > 0:
        # stratified-ish: take first k per tamper type
        for tt, g in df.groupby("tamper_type"):
            for _, r in g.head(max(1, need // max(1, df["tamper_type"].nunique()))).iterrows():
                if len(examples) >= args.save_examples:
                    break
                tid = r["trial_id"]
                tr = trial_by_id.get(tid, {})
                ro = roll_by_id.get(tid, {})
                examples.append({
                    "trial_id": tid,
                    "tamper_type": tr.get("tamper_type"),
                    "tamper_pct": tr.get("tamper_pct"),
                    "is_tampered": tr.get("is_tampered"),
                    "dataset": tr.get("dataset"),
                    "prompt": tr.get("prompt"),
                    "cot_half": tr.get("cot_half"),
                    "tampered_half": tr.get("tampered_half"),
                    "self_report_flag": ro.get("self_report_flag"),
                    "self_report_parse_ok": ro.get("self_report_parse_ok"),
                    "boxed_answer": ro.get("boxed_answer"),
                    "assistant_text": ro.get("assistant_text"),
                    "full_text": ro.get("full_text"),
                })
            if len(examples) >= args.save_examples:
                break

    write_jsonl(out_dir / "examples_prefill.jsonl", examples)
    print(f"Saved {len(examples)} examples to {out_dir / 'examples_prefill.jsonl'}")

    # Train activation probe
    X_train, y_train = X_act[train_idx], y_vec[train_idx]
    X_test, y_test = X_act[test_idx], y_vec[test_idx]

    clf_act = fit_logreg_classifier(X_train, y_train)
    met_act = eval_classifier(clf_act, X_test, y_test)

    # Text-only baseline: NLL
    X_nll = df["nll_prefill"].to_numpy().reshape(-1, 1)
    clf_nll = fit_logreg_classifier(X_nll[train_idx], y_train)
    met_nll = eval_classifier(clf_nll, X_nll[test_idx], y_test)

    # Paper-style curves (tampered only): detection rate & answer correctness by (type, pct)
    curves = {}
    if (not args.skip_self_report) and ("self_report_modified" in df.columns):
        tampered = df[df["is_tampered"] == 1].copy()
        for (tt, pct), g in tampered.groupby(["tamper_type", "tamper_pct"], dropna=False):
            curves[str((tt, pct))] = {
                "n": len(g),
                "tamper_detection_rate": float(g["self_report_modified"].mean()),
                "parse_ok_rate": float(g["self_report_parse_ok"].mean()) if "self_report_parse_ok" in g else None,
                "answer_correct_rate": float(g["answer_correct"].dropna().mean()) if "answer_correct" in g else None,
            }

    metrics = {
        "activation_probe": met_act,
        "text_nll_baseline": met_nll,
        "paper_style_curves": curves,
        "n_trials": len(trials),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "tamper_types": args.tamper_types,
        "tamper_pcts": args.tamper_pcts,
        "layers": args.layers,
        "model": args.model,
    }

    (out_dir / "metrics_prefill.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    joblib.dump(clf_act, out_dir / "clf_activation_probe_prefill.joblib")
    joblib.dump(clf_nll, out_dir / "clf_nll_prefill.joblib")


if __name__ == "__main__":
    main()
