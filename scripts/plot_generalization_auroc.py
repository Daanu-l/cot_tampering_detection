import argparse
import json
from pathlib import Path
from typing import Any

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import numpy as np

TAMPER_ORDER = ["token_delete", "token_insert", "sentence_remove", "sentence_paraphrase"]
TAMPER_LABELS = {
    "token_delete": "token_delete",
    "token_insert": "token_insert",
    "sentence_remove": "sentence_remove",
    "sentence_paraphrase": "sentence_paraphrase",
}


def _get_auroc(entry: dict[str, Any]) -> float | None:
    if entry is None:
        return None
    v = entry.get("auroc")
    return None if v is None else float(v)


def _barplot_auroc(
    ax, title: str, x_labels: list[str], auroc_act: list[float | None], auroc_nll: list[float | None], ylim: tuple[float, float] = (0.5, 1.0)
):
    # White background, no grid
    ax.set_facecolor("white")
    ax.grid(False)

    x = np.arange(len(x_labels))
    width = 0.36

    # Replace None with NaN so plotting works
    y1 = np.array([np.nan if v is None else v for v in auroc_act], dtype=float)
    y2 = np.array([np.nan if v is None else v for v in auroc_nll], dtype=float)

    # Colors: keep simple and readable
    b1 = ax.bar(x - width / 2, y1, width, label="Activation probe", color="#1f77b4")
    b2 = ax.bar(x + width / 2, y2, width, label="Text NLL baseline", color="#ff7f0e")

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("AUROC")
    ax.set_ylim(*ylim)

    # Annotate values above bars (bigger font), but don’t crash on NaNs
    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,  # bigger
                fontweight="bold",
                clip_on=False,
            )

    annotate(b1)
    annotate(b2)

    # Legend outside plot area
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)


def _heatmap_auroc(ax, title: str, mat: np.ndarray, row_labels: list[str], col_labels: list[str], cmap=None, vmin: float = 0.90, vmax: float = 1.00):
    ax.set_facecolor("white")
    ax.grid(False)

    # Default: smooth monotone green that does NOT start at white
    if cmap is None:
        base = plt.get_cmap("Greens")
        cmap = LinearSegmentedColormap.from_list(
            "Greens_no_white",
            base(np.linspace(0.25, 1.0, 256)),  # start at 25% to avoid near-white
        )

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Test tamper type")
    ax.set_ylabel("Train tamper type")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                txt = "—"
                color = "black"
            else:
                txt = f"{v:.3f}"
                color = "white" if v >= (vmin + 0.7 * (vmax - vmin)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--json_name", type=str, default="generalization_prefill.json")
    ap.add_argument("--out_prefix", type=str, default="prefill")
    ap.add_argument("--heatmap_vmin", type=float, default=0.90, help="Lower bound for heatmap color scaling (clips lower values).")
    ap.add_argument("--heatmap_vmax", type=float, default=1.00, help="Upper bound for heatmap color scaling.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gen_path = run_dir / args.json_name
    data = json.loads(gen_path.read_text(encoding="utf-8"))

    # -------------------------
    # 1) Cross-domain barplot
    # -------------------------
    cross_domain = data.get("cross_domain", {})
    # Prefer consistent order: math->mmlu then mmlu->math if present
    domain_keys = ["train_math__test_mmlu", "train_mmlu__test_math"]
    available = [k for k in domain_keys if k in cross_domain]
    # fallback to whatever exists
    if not available:
        available = sorted(cross_domain.keys())

    x_labels = []
    auroc_act = []
    auroc_nll = []
    for k in available:
        # Pretty label
        k2 = k.replace("train_", "").replace("__test_", "→")
        x_labels.append(k2)

        entry = cross_domain[k]
        auroc_act.append(_get_auroc(entry.get("activation_probe", {})))
        auroc_nll.append(_get_auroc(entry.get("text_nll_baseline", {})))

    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor="white")
    _barplot_auroc(ax=ax, title="Cross-domain generalization (AUROC)", x_labels=x_labels, auroc_act=auroc_act, auroc_nll=auroc_nll, ylim=(0.0, 1.10))
    plt.tight_layout()
    out1 = run_dir / f"{args.out_prefix}_cross_domain_auroc.png"
    plt.savefig(out1, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out1}")

    # ---------------------------------
    # 2) Leave-one-tamper-out barplot
    # ---------------------------------
    loo = data.get("leave_one_tamper_out", {})
    # Order tokens first then sentences, matching your preference
    loo_keys = [f"heldout_{t}" for t in TAMPER_ORDER if f"heldout_{t}" in loo]
    # fallback
    if not loo_keys:
        loo_keys = sorted(loo.keys())

    x_labels = []
    auroc_act = []
    auroc_nll = []
    for k in loo_keys:
        held = k.replace("heldout_", "")
        x_labels.append(TAMPER_LABELS.get(held, held))

        entry = loo[k]
        auroc_act.append(_get_auroc(entry.get("activation_probe", {})))
        auroc_nll.append(_get_auroc(entry.get("text_nll_baseline", {})))

    fig, ax = plt.subplots(figsize=(10.5, 4.2), facecolor="white")
    _barplot_auroc(
        ax=ax, title="Leave-one-tamper-type-out generalization (AUROC)", x_labels=x_labels, auroc_act=auroc_act, auroc_nll=auroc_nll, ylim=(0.0, 1.10)
    )
    plt.tight_layout()
    out2 = run_dir / f"{args.out_prefix}_leave_one_tamper_out_auroc.png"
    plt.savefig(out2, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out2}")

    # ---------------------------------
    # 3) Cross-tamper heatmap (AUROC) — Activation probe only
    # ---------------------------------
    cross_tamper = data.get("cross_tamper", {})

    # Use preferred ordering: tokens first then sentences.
    tts = [t for t in TAMPER_ORDER if any(f"train_{t}__test_" in k for k in cross_tamper.keys())]
    if not tts:
        inferred = set()
        for k in cross_tamper.keys():
            if k.startswith("train_") and "__test_" in k:
                a = k.split("__test_")[0].replace("train_", "")
                b = k.split("__test_")[1]
                inferred.add(a)
                inferred.add(b)
        tts = sorted(list(inferred))

    n = len(tts)
    mat_act = np.full((n, n), np.nan, dtype=float)

    keyfmt = "train_{train}__test_{test}"
    for i, tr in enumerate(tts):
        for j, te in enumerate(tts):
            k = keyfmt.format(train=tr, test=te)
            if k not in cross_tamper:
                continue
            entry = cross_tamper[k]
            mat_act[i, j] = _get_auroc(entry.get("activation_probe", {}))

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.2), facecolor="white")
    fig.suptitle("Cross-tamper generalization (AUROC)", fontsize=15, y=1.02)

    _heatmap_auroc(
        ax=ax,
        title="Activation probe",
        mat=mat_act,
        row_labels=[TAMPER_LABELS.get(t, t) for t in tts],
        col_labels=[TAMPER_LABELS.get(t, t) for t in tts],
        cmap=None,  # uses the smooth non-white Greens default in _heatmap_auroc
        vmin=args.heatmap_vmin,
        vmax=args.heatmap_vmax,
    )

    plt.tight_layout()
    out3 = run_dir / f"{args.out_prefix}_cross_tamper_auroc_heatmap.png"
    plt.savefig(out3, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out3}")

    print("Done.")


if __name__ == "__main__":
    main()
