"""PART B — The Retriever (two-stage retrieve-and-rerank).

    Stage 1 (recall, cheap, scalable):
        Encode the query with FashionCLIP, ANN search Chroma -> top-N candidates.
    Stage 2 (precision, expensive, bounded):
        Re-score the N candidates with BLIP-ITM cross-attention.
        final = alpha * clip_cosine + (1 - alpha) * blip_itm_prob
    Return top-k by the fused score.

The re-ranker only sees N (~50) images regardless of database size, so precision
improves without sacrificing scalability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import config
from .backbone import get_backbone
from .indexer import get_collection
from .reranker import get_reranker


@dataclass
class RetrievalResult:
    filename: str
    score: float          # fused final score
    clip_score: float     # stage-1 cosine (rescaled to 0..1)
    rerank_score: float   # stage-2 BLIP-ITM P(match)
    stage1_rank: int      # position before reranking (to show the reordering)


class TwoStageRetriever:
    def __init__(self, top_n: int = config.STAGE1_TOPN, alpha: float = config.RERANK_ALPHA):
        self.top_n = top_n
        self.alpha = alpha
        self.backbone = get_backbone()
        self.collection = get_collection()

    def _stage1(self, query: str):
        q = self.backbone.encode_texts([query])[0].tolist()
        res = self.collection.query(query_embeddings=[q], n_results=self.top_n)
        ids = res["ids"][0]
        # Chroma cosine distance -> similarity in [0,1]
        clip_scores = [1.0 - d / 2.0 for d in res["distances"][0]]
        return ids, clip_scores

    def search(self, query: str, k: int = 5, rerank: bool = True) -> List[RetrievalResult]:
        ids, clip_scores = self._stage1(query)

        if not rerank:
            return [
                RetrievalResult(f, s, s, float("nan"), i)
                for i, (f, s) in enumerate(zip(ids[:k], clip_scores[:k]))
            ]

        paths = [config.IMAGE_DIR / f for f in ids]
        itm = get_reranker().score(query, paths)

        # Scale-invariant fusion: CLIP cosine and BLIP-ITM live on very different
        # scales (ITM match-probs are tiny/compressed for out-of-domain images),
        # so we min-max normalize each score across the candidate set before the
        # alpha blend. This lets the cross-encoder's *relative* ordering actually
        # move results, instead of being swamped by CLIP's larger magnitudes.
        def _mn(xs):
            lo, hi = min(xs), max(xs)
            return [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in xs]

        c_n, r_n = _mn(clip_scores), _mn(itm)
        fused = [self.alpha * c + (1 - self.alpha) * r for c, r in zip(c_n, r_n)]

        order = sorted(range(len(ids)), key=lambda i: -fused[i])
        return [
            RetrievalResult(
                filename=ids[i],
                score=float(fused[i]),
                clip_score=float(clip_scores[i]),
                rerank_score=float(itm[i]),
                stage1_rank=i,
            )
            for i in order[:k]
        ]
