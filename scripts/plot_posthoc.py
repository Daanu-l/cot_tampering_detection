import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--json_name", type=str, default="metrics_posthoc.json")
    ap.add_argument("--out_name", type=str, default="overall_posthoc_three_way.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data = json.loads((run_dir / args.json_name).read_text(encoding="utf-8"))

    nll = data.get("text_nll_baseline", {})
    vec = data.get("posthoc_activation_vector_probe", {})
    sim = data.get("posthoc_activation_similarity_probe", {})

    metrics = ["balanced_acc", "auroc"]
    labels = ["Balanced accuracy", "AUROC"]

    def get_vals(d):
        vals = [d.get(m, None) for m in metrics]
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=float)

    nll_vals = get_vals(nll)
    vec_vals = get_vals(vec)
    sim_vals = get_vals(sim)

    x = np.arange(len(metrics))
    width = 0.24  # 3 bars per group

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(False)

    # Match your palette style: use the same two colors plus one more consistent one
    b1 = ax.bar(x - width, nll_vals, width, label="Text NLL baseline", color="#ff7f0e")
    b2 = ax.bar(x, vec_vals, width, label="Activation vector probe", color="#1f77b4")
    b3 = ax.bar(x + width, sim_vals, width, label="Activation similarity probe", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score")
    # ax.set_title("Post-hoc detection performance")
    ax.set_title("Ablation: Post-hoc performance at 2% tampering")

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", clip_on=False)

    annotate(b1)
    annotate(b2)
    annotate(b3)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout()

    out_path = run_dir / args.out_name
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
