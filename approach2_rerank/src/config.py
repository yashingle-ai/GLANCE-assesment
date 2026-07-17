"""Central configuration for Approach 2 (Two-Stage Retrieve-and-Rerank)."""
from __future__ import annotations

from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
APPROACH_DIR = SRC_DIR.parent
REPO_ROOT = APPROACH_DIR.parent

IMAGE_DIR = REPO_ROOT / "dataset" / "images_1000"
INDEX_DIR = APPROACH_DIR / "artifacts"
CHROMA_DIR = INDEX_DIR / "chroma"
EMB_CACHE = INDEX_DIR / "image_embeddings.npz"
RESULTS_DIR = APPROACH_DIR / "results"

COLLECTION_NAME = "fashion_fashionclip"

# Stage-1 recall backbone (fast bi-encoder, ANN-searchable).
BACKBONE_MODEL = "patrickjohncyh/fashion-clip"
# Stage-2 precision re-ranker: BLIP image-text matching (cross-attention).
RERANKER_MODEL = "Salesforce/blip-itm-base-coco"

BATCH_SIZE = 32

# Two-stage knobs
STAGE1_TOPN = 50    # candidates pulled by ANN recall
RERANK_ALPHA = 0.5  # final = alpha*clip_cosine + (1-alpha)*blip_itm_prob


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


for _d in (INDEX_DIR, CHROMA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
