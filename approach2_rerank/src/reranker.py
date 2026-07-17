"""Stage-2 re-ranker — BLIP Image-Text Matching (the precision lever).

A CLIP bi-encoder scores image and text *independently* (one vector each), so it
cannot reason about how they relate — this is why context ("in a modern office")
and binding ("red tie / white shirt") leak.

BLIP-ITM is a CROSS-encoder: it feeds the image patches and the query tokens
through joint cross-attention and outputs P(match). That joint view resolves
context and attribute binding that the bi-encoder blurs. It is ~100x heavier per
pair, so we only ever run it on the stage-1 top-N candidates, not the whole DB —
which is exactly what keeps the system scalable to millions of images.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor

from .config import RERANKER_MODEL, get_device


class BlipReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str | None = None):
        self.device = device or get_device()
        self.model = (
            BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device).eval()
        )
        self.processor = BlipProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(self, query: str, image_paths: List[Path | str]) -> List[float]:
        """P(image matches query) for each candidate image."""
        probs = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            inputs = self.processor(images=img, text=query, return_tensors="pt").to(
                self.device
            )
            logits = self.model(**inputs, use_itm_head=True)[0]  # (1, 2)
            probs.append(float(torch.softmax(logits, dim=1)[0, 1]))
        return probs


@lru_cache(maxsize=1)
def get_reranker() -> BlipReranker:
    return BlipReranker()
