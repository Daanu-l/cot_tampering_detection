import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--metric", type=str, default="auroc", choices=["auroc", "balanced_acc"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "layer_sweep_prefill.csv")

    df = df.sort_values(by=args.metric, ascending=False)

    plt.figure()
    plt.bar(df["layer"], df[args.metric])
    plt.xticks(rotation=45)
    plt.ylabel(args.metric)
    plt.title(f"Single-layer probe ({args.metric})")
    plt.tight_layout()

    out = run_dir / f"best_single_layer_{args.metric}.png"
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
