"""Evaluation helpers: 5 assessment queries, result grids, and a
before/after comparison that visualizes what stage-2 reranking changes."""
from __future__ import annotations

from . import config

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def show_results(query, results, cols: int = 5, save_as: str | None = None):
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, results):
        img = Image.open(config.IMAGE_DIR / r.filename).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{r.score:.3f}\n(clip {r.clip_score:.2f} | itm {r.rerank_score:.2f})",
                     fontsize=8)
    fig.suptitle(query, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_as:
        out = config.RESULTS_DIR / save_as
        fig.savefig(out, bbox_inches="tight", dpi=110)
        print(f"[eval] saved {out}")
    return fig


def compare_stage1_vs_reranked(retriever, query, k: int = 5):
    """Side-by-side: CLIP-only recall vs. two-stage reranked."""
    return {
        "stage1_only": retriever.search(query, k=k, rerank=False),
        "reranked": retriever.search(query, k=k, rerank=True),
    }


def run_all(retriever, k: int = 5, save: bool = True):
    all_results = {}
    for i, q in enumerate(EVAL_QUERIES, 1):
        res = retriever.search(q, k=k, rerank=True)
        all_results[q] = res
        if save:
            show_results(q, res, save_as=f"query_{i}.png")
    return all_results
