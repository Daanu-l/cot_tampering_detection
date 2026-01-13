import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

KEY_RE = re.compile(r"\('(?P<tt>[^']+)',\s*np\.float64\((?P<pct>nan|[-+]?\d*\.?\d+)\)\)")

ORDER = ["token_delete", "token_insert", "sentence_remove", "sentence_paraphrase"]


def parse_key(k: str):
    """
    Keys look like: "('token_delete', np.float64(20.0))" or nan for paraphrase.
    Returns (tamper_type, pct_or_none).
    """
    m = KEY_RE.match(k.strip())
    if not m:
        # fallback: try to recover just the first quoted segment
        tt = k.strip()
        return tt, None
    tt = m.group("tt")
    pct_s = m.group("pct")
    if pct_s == "nan":
        return tt, None
    return tt, float(pct_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--json_name", type=str, default="metrics_prefill.json")
    ap.add_argument("--out_name", type=str, default="paper_style_curves.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data = json.loads((run_dir / args.json_name).read_text(encoding="utf-8"))
    curves = data.get("paper_style_curves", {})

    rows = []
    for k, v in curves.items():
        tt, pct = parse_key(k)
        rows.append({
            "tamper_type": tt,
            "pct": pct,
            "n": v.get("n", None),
            "tamper_detection_rate": v.get("tamper_detection_rate", None),
            "answer_correct_rate": v.get("answer_correct_rate", None),
        })

    # Sort in preferred order
    def sort_key(r):
        tt = r["tamper_type"]
        idx = ORDER.index(tt) if tt in ORDER else 999
        pct = r["pct"]
        pct_val = -1 if pct is None else pct
        return (idx, pct_val)

    rows.sort(key=sort_key)

    labels = []
    det = []
    acc = []
    for r in rows:
        tt = r["tamper_type"]
        pct = r["pct"]
        lab = f"{tt}" if pct is None else f"{tt} ({pct:.0f}%)"
        labels.append(lab)

        det.append(np.nan if r["tamper_detection_rate"] is None else float(r["tamper_detection_rate"]))
        acc.append(np.nan if r["answer_correct_rate"] is None else float(r["answer_correct_rate"]))

    det = np.array(det, dtype=float)
    acc = np.array(acc, dtype=float)

    x = np.arange(len(labels))
    width = 0.40

    fig, ax = plt.subplots(figsize=(10.8, 4.4), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(False)

    # Always plot detection rate (green)
    b1 = ax.bar(x - width / 2, det, width, label="Self-report CoT modification rate", color="#2ca02c")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.10)

    # Plot answer correctness only if any non-NaN exists
    if np.isfinite(acc).any():
        b2 = ax.bar(x + width / 2, acc, width, label="Answer Correctness", color="#1f77b4")
        # annotate both
        bars_to_annotate = [b1, b2]
    else:
        bars_to_annotate = [b1]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)  # , rotation=30, ha="right")
    ax.set_title("Self-report CoT modification vs Answer Correctness under tampering")
    ax.legend(frameon=False, loc="upper right")

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", clip_on=False)

    for b in bars_to_annotate:
        annotate(b)

    plt.tight_layout()
    out_path = run_dir / args.out_name
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
