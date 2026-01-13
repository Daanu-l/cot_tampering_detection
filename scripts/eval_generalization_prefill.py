import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_lr(X, y):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
    ]).fit(X, y)


def eval_lr(clf, X, y):
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None,
        "n": len(y),
        "pos_rate": float(y.mean()),
    }


def load_layer_slices(run_dir: Path):
    p = run_dir / "layer_slices.json"
    if not p.exists():
        return None
    js = json.loads(p.read_text(encoding="utf-8"))
    # normalize keys to str
    slices = {str(k): v for k, v in js["slices"].items()}
    layers = [str(x) for x in js["layers"]]
    return {"layers": layers, "slices": slices, "d_per_layer": js.get("d_per_layer")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--min_n", type=int, default=40, help="Skip train/test subsets smaller than this.")
    ap.add_argument("--sweep_layers", action="store_true", help="If set and layer_slices.json exists, train/eval a probe per layer slice.")
    ap.add_argument(
        "--best_metric", type=str, default="auroc", choices=["auroc", "balanced_acc"], help="Metric used to pick best single layer in the summary."
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    X = np.load(run_dir / "X_act_prefill_fp16.npy").astype(np.float32)
    y = np.load(run_dir / "y_prefill.npy").astype(int)
    train_idx = np.load(run_dir / "train_idx.npy").astype(int)
    test_idx = np.load(run_dir / "test_idx.npy").astype(int)

    df = pd.read_csv(run_dir / "features_prefill.csv")
    order = (run_dir / "trial_id_order.txt").read_text(encoding="utf-8").splitlines()
    df = df.set_index("trial_id").loc[order].reset_index()

    # NLL baseline feature
    nll = df["nll_prefill"].to_numpy().reshape(-1, 1).astype(np.float32)

    tamper_types = sorted(df["tamper_type"].dropna().unique().tolist())
    datasets = sorted(df["dataset"].dropna().unique().tolist())

    # Helper: select subset within train/test partitions
    def subset(mask, partition_idx):
        idx = partition_idx[mask[partition_idx]]
        return idx

    # ---------- Base (concatenated) results ----------
    results = {
        "cross_tamper": {},
        "leave_one_tamper_out": {},
        "cross_domain": {},
        "notes": {"train_idx_len": len(train_idx), "test_idx_len": len(test_idx), "tamper_types": tamper_types, "datasets": datasets},
    }

    # Cross-tamper generalization
    for train_tt in tamper_types:
        for test_tt in tamper_types:
            key = f"train_{train_tt}__test_{test_tt}"
            train_mask = df["tamper_type"].values == train_tt
            test_mask = df["tamper_type"].values == test_tt

            tr = subset(train_mask, train_idx)
            te = subset(test_mask, test_idx)

            if len(tr) < args.min_n or len(te) < args.min_n:
                continue

            clf_act = fit_lr(X[tr], y[tr])
            clf_nll = fit_lr(nll[tr], y[tr])

            results["cross_tamper"][key] = {"activation_probe": eval_lr(clf_act, X[te], y[te]), "text_nll_baseline": eval_lr(clf_nll, nll[te], y[te])}

    # Leave-one-out tamper type
    for heldout in tamper_types:
        key = f"heldout_{heldout}"
        train_mask = df["tamper_type"].values != heldout
        test_mask = df["tamper_type"].values == heldout

        tr = subset(train_mask, train_idx)
        te = subset(test_mask, test_idx)

        if len(tr) < args.min_n or len(te) < args.min_n:
            continue

        clf_act = fit_lr(X[tr], y[tr])
        clf_nll = fit_lr(nll[tr], y[tr])

        results["leave_one_tamper_out"][key] = {
            "activation_probe": eval_lr(clf_act, X[te], y[te]),
            "text_nll_baseline": eval_lr(clf_nll, nll[te], y[te]),
        }

    # Cross-domain generalization
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds == test_ds:
                continue
            key = f"train_{train_ds}__test_{test_ds}"

            train_mask = df["dataset"].values == train_ds
            test_mask = df["dataset"].values == test_ds

            tr = subset(train_mask, train_idx)
            te = subset(test_mask, test_idx)

            if len(tr) < args.min_n or len(te) < args.min_n:
                continue

            clf_act = fit_lr(X[tr], y[tr])
            clf_nll = fit_lr(nll[tr], y[tr])

            results["cross_domain"][key] = {"activation_probe": eval_lr(clf_act, X[te], y[te]), "text_nll_baseline": eval_lr(clf_nll, nll[te], y[te])}

    out_path = run_dir / "generalization_prefill.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # ---------- Per-layer sweep (optional) ----------
    layer_info = load_layer_slices(run_dir)
    if args.sweep_layers and layer_info is not None:
        layers = layer_info["layers"]
        slices = layer_info["slices"]

        layer_rows = []
        per_layer = {}

        # Overall (same split) per-layer probe, plus NLL baseline for context
        clf_nll_full = fit_lr(nll[train_idx], y[train_idx])
        nll_overall = eval_lr(clf_nll_full, nll[test_idx], y[test_idx])

        for layer in layers:
            sl = slices[layer]
            Xl = X[:, sl["start"] : sl["end"]]

            clf_l = fit_lr(Xl[train_idx], y[train_idx])
            met_l = eval_lr(clf_l, Xl[test_idx], y[test_idx])

            per_layer[layer] = {"overall": met_l, "nll_overall": nll_overall, "slice": sl}

            layer_rows.append({
                "layer": layer,
                "balanced_acc": met_l["balanced_acc"],
                "auroc": met_l["auroc"],
                "n_test": met_l["n"],
                "nll_balanced_acc": nll_overall["balanced_acc"],
                "nll_auroc": nll_overall["auroc"],
            })

        # pick best layer
        def key_fn(r):
            v = r.get(args.best_metric)
            return -1.0 if v is None else float(v)

        best = max(layer_rows, key=key_fn)
        summary = {
            "best_metric": args.best_metric,
            "best_layer": best["layer"],
            "best_layer_metrics": {k: best[k] for k in ["balanced_acc", "auroc", "n_test"]},
            "layers_tested": layers,
        }

        (run_dir / "layer_sweep_prefill.json").write_text(json.dumps({"per_layer": per_layer, "summary": summary}, indent=2), encoding="utf-8")

        pd.DataFrame(layer_rows).sort_values(by=args.best_metric, ascending=False).to_csv(run_dir / "layer_sweep_prefill.csv", index=False)

        print(f"Wrote {run_dir / 'layer_sweep_prefill.json'}")
        print(f"Wrote {run_dir / 'layer_sweep_prefill.csv'}")
        print(f"Best single layer by {args.best_metric}: {best['layer']} (auroc={best['auroc']}, bal_acc={best['balanced_acc']})")
    elif args.sweep_layers:
        print("No layer_slices.json found; skipping per-layer sweep.")


if __name__ == "__main__":
    main()
