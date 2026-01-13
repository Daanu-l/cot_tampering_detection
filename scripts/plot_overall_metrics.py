import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--json_name", type=str, default="metrics_prefill.json")
    ap.add_argument("--out_name", type=str, default="overall_probe_vs_nll.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data = json.loads((run_dir / args.json_name).read_text(encoding="utf-8"))

    act = data.get("activation_probe", {})
    nll = data.get("text_nll_baseline", {})

    metrics = ["balanced_acc", "auroc"]
    labels = ["Balanced accuracy", "AUROC"]

    act_vals = [act.get(m, None) for m in metrics]
    nll_vals = [nll.get(m, None) for m in metrics]

    # Convert None to NaN for plotting
    act_vals = np.array([np.nan if v is None else float(v) for v in act_vals], dtype=float)
    nll_vals = np.array([np.nan if v is None else float(v) for v in nll_vals], dtype=float)

    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(False)

    # Green for activation, gray for baseline
    b1 = ax.bar(x - width / 2, act_vals, width, label="Activation probe", color="#1f77b4")
    b2 = ax.bar(x + width / 2, nll_vals, width, label="Text NLL baseline", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Overall detection performance")
    # ax.set_title("Ablation: Detection performance at 2% tampering")

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold", clip_on=False)

    annotate(b1)
    annotate(b2)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout()

    out_path = run_dir / args.out_name
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
