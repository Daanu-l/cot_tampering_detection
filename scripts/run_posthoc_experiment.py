"""
Run the *post-hoc displayed-trace tampering* experiment.

We assume a clean reference CoT exists, but the displayed CoT is edited.
Detectors include:
- Text NLL baseline on the displayed trace
- Activation-vector linear probe
- Reference-based similarity features (LR on similarity scalars)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from cot_integrity.data import load_math500, load_mmlu
from cot_integrity.detectors import eval_classifier, fit_logreg_classifier
from cot_integrity.features import extract_posthoc_actvec, extract_posthoc_features
from cot_integrity.modeling import generate_assistant_text, load_model
from cot_integrity.prompts import messages_base_solve, messages_llm_judge, messages_paraphrase_sentence
from cot_integrity.tamper import tamper_dispatch
from cot_integrity.utils import ensure_dir, read_jsonl, set_seed, stable_hash, write_jsonl


def make_paraphrase_fn(bundle, paraphrase_temperature: float = 0.7, max_new_tokens: int = 96):
    def paraphrase(sentence: str) -> str:
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
        return assistant.strip().split("\n")[0].strip().strip('"')

    return paraphrase


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--n_math", type=int, default=60)
    ap.add_argument("--n_mmlu", type=int, default=60)
    ap.add_argument("--mmlu_subjects", type=str, nargs="*", default=None)

    ap.add_argument("--tamper_types", type=str, nargs="+", default=["token_delete", "token_insert", "sentence_remove", "sentence_paraphrase"])

    ap.add_argument("--layer", type=int, default=-1, help="Layer index for sequence hidden states, usually -1")
    ap.add_argument("--layers", type=int, nargs="+", default=[-1], help="Layers for activation-vector probe (like prefill).")

    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--max_new_tokens_base", type=int, default=1024)
    ap.add_argument("--tamper_pcts", type=int, nargs="+", default=[20], help="Match prefill: pct for token_delete/token_insert/sentence_remove.")
    ap.add_argument(
        "--prefill_out_dir",
        type=str,
        default=None,
        help="If set, reuse baseline_generations.jsonl from this prefill run instead of generating new CoTs.",
    )
    ap.add_argument(
        "--use_cot_half", action="store_true", help="Use cot_half (like prefill tampering surface) instead of cot_full for posthoc trials."
    )

    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    bundle = load_model(args.model, device_map=args.device_map, torch_dtype=args.torch_dtype)

    # Load examples
    examples = []
    if args.n_math > 0:
        examples += load_math500(args.n_math, seed=args.seed)
    if args.n_mmlu > 0:
        examples += load_mmlu(args.n_mmlu, seed=args.seed, subjects=args.mmlu_subjects)

    if args.prefill_out_dir is None:
        raise ValueError("Set --prefill_out_dir to reuse prefill CoTs (recommended).")

    prefill_dir = Path(args.prefill_out_dir)
    base_path = prefill_dir / "baseline_generations.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"Could not find {base_path}")

    baseline_rows = read_jsonl(base_path)

    # Build a baseline dict in the posthoc format
    baseline = {}
    for r in baseline_rows:
        cot = r["cot_half"] if args.use_cot_half else r["cot_full"]
        if cot is None or len(cot.strip()) < 5:
            continue
        baseline[r["uid"]] = {"uid": r["uid"], "dataset": r["dataset"], "subject": r.get("subject"), "prompt": r["prompt"], "cot": cot}

    paraphrase_fn = make_paraphrase_fn(bundle) if ("sentence_paraphrase" in args.tamper_types) else None
    rng = random.Random(args.seed)

    # Build post-hoc trials: displayed trace may be tampered
    trials = []
    sweep_types = {"token_delete", "token_insert", "sentence_remove"}
    for uid, b in baseline.items():
        clean = b["cot"]
        for tt in args.tamper_types:
            if tt in sweep_types:
                for pct in args.tamper_pcts:
                    # control trial
                    trials.append({
                        "trial_id": stable_hash(f"{uid}:{tt}:{pct}:control"),
                        "base_uid": uid,
                        "dataset": b["dataset"],
                        "subject": b.get("subject"),
                        "prompt": b["prompt"],
                        "tamper_type": tt,
                        "tamper_pct": int(pct),
                        "is_tampered": 0,
                        "clean_cot": clean,
                        "displayed_cot": clean,
                        "tamper_meta": {"type": tt, "pct": int(pct), "control": True},
                    })
                    # tampered (skip pct=0)
                    if pct > 0:
                        tam = tamper_dispatch(tamper_type=tt, text=clean, tokenizer=bundle.tokenizer, rng=rng, pct=int(pct), paraphrase_fn=None)
                        trials.append({
                            "trial_id": stable_hash(f"{uid}:{tt}:{pct}:tampered"),
                            "base_uid": uid,
                            "dataset": b["dataset"],
                            "subject": b.get("subject"),
                            "prompt": b["prompt"],
                            "tamper_type": tt,
                            "tamper_pct": int(pct),
                            "is_tampered": 1,
                            "clean_cot": clean,
                            "displayed_cot": tam.tampered_text,
                            "tamper_meta": tam.meta,
                        })
            else:
                trials.append({
                    "trial_id": stable_hash(f"{uid}:{tt}:control"),
                    "base_uid": uid,
                    "dataset": b["dataset"],
                    "subject": b.get("subject"),
                    "prompt": b["prompt"],
                    "tamper_type": tt,
                    "tamper_pct": None,
                    "is_tampered": 0,
                    "clean_cot": clean,
                    "displayed_cot": clean,
                    "tamper_meta": {"type": tt, "control": True},
                })

                tam = tamper_dispatch(tamper_type=tt, text=clean, tokenizer=bundle.tokenizer, rng=rng, pct=None, paraphrase_fn=paraphrase_fn)
                trials.append({
                    "trial_id": stable_hash(f"{uid}:{tt}:tampered"),
                    "base_uid": uid,
                    "dataset": b["dataset"],
                    "subject": b.get("subject"),
                    "prompt": b["prompt"],
                    "tamper_type": tt,
                    "is_tampered": 1,
                    "clean_cot": clean,
                    "displayed_cot": tam.tampered_text,
                    "tamper_meta": tam.meta,
                })

    write_jsonl(out_dir / "trials_posthoc.jsonl", trials)

    # Group split by base_uid
    groups = np.array([t["base_uid"] for t in trials])
    y = np.array([t["is_tampered"] for t in trials], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    (train_idx, test_idx) = next(splitter.split(np.zeros(len(trials)), y, groups=groups))

    # Extract features (posthoc similarity + NLL)
    feat_rows = []
    X = []
    X_sim = []
    X_act = []
    y_vec = []

    for t in tqdm(trials, desc="Extracting post-hoc features"):
        msgs = messages_base_solve(t["prompt"])

        feats = extract_posthoc_features(
            bundle=bundle,
            messages=msgs,
            clean_prefill_text="<think>\n" + t["clean_cot"].strip() + "\n</think>\n",
            displayed_prefill_text="<think>\n" + t["displayed_cot"].strip() + "\n</think>\n",
            layer_index=args.layer,
        )
        x_act = extract_posthoc_actvec(
            bundle=bundle, messages=msgs, displayed_prefill_text="<think>\n" + t["displayed_cot"].strip() + "\n</think>\n", layers=args.layers
        )
        X_act.append(x_act)
        row = {
            "trial_id": t["trial_id"],
            "base_uid": t["base_uid"],
            "dataset": t["dataset"],
            "subject": t.get("subject"),
            "tamper_type": t["tamper_type"],
            "is_tampered": int(t["is_tampered"]),
            **feats,
        }

        if not args.skip_judge:
            judge = maybe_llm_judge(bundle, t["prompt"], t["displayed_cot"])
            row["llm_judge_modified"] = None if judge is None else int(judge)

        feat_rows.append(row)
        X_sim.append([feats["sim_mean"], feats["sim_p10"], feats["match_frac"], feats["seq_ratio"], feats["len_diff"]])
        y_vec.append(int(t["is_tampered"]))

    X_act_np = np.stack(X_act, axis=0).astype(np.float16)
    np.save(out_dir / "X_act_posthoc_fp16.npy", X_act_np)
    np.save(out_dir / "y_posthoc.npy", np.array(y_vec, dtype=np.int64))

    df = pd.DataFrame(feat_rows)
    df.to_csv(out_dir / "features_posthoc.csv", index=False)

    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    X = np.array(X_sim, dtype=np.float32)
    y_vec = np.array(y_vec, dtype=int)

    # Train on similarity features
    clf_sim = fit_logreg_classifier(X[train_idx], y_vec[train_idx])
    met_sim = eval_classifier(clf_sim, X[test_idx], y_vec[test_idx])

    # Text-only baseline: NLL_displayed as 1D feature
    X_nll = df["nll_displayed"].to_numpy().reshape(-1, 1)
    clf_nll = fit_logreg_classifier(X_nll[train_idx], y_vec[train_idx])
    met_nll = eval_classifier(clf_nll, X_nll[test_idx], y_vec[test_idx])

    X_act = np.stack(X_act, axis=0).astype(np.float32)
    clf_act = fit_logreg_classifier(X_act[train_idx], y_vec[train_idx])
    met_act = eval_classifier(clf_act, X_act[test_idx], y_vec[test_idx])
    joblib.dump(clf_act, out_dir / "clf_activation_probe_posthoc_actvec.joblib")

    metrics = {
        "posthoc_activation_similarity_probe": met_sim,
        "text_nll_baseline": met_nll,
        "n_trials": len(trials),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "tamper_types": args.tamper_types,
        "layer": args.layer,
        "model": args.model,
        "posthoc_activation_vector_probe": met_act,
        "layers": args.layers,
    }

    (out_dir / "metrics_posthoc.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    joblib.dump(clf_sim, out_dir / "clf_activation_probe_posthoc.joblib")
    joblib.dump(clf_nll, out_dir / "clf_nll_posthoc.joblib")


if __name__ == "__main__":
    main()
