# Approach 2 — Two-Stage Retrieve-and-Rerank

Text-to-image fashion retrieval that maximizes **precision** with a two-stage pipeline:
fast FashionCLIP recall, then a BLIP image-text-matching **cross-encoder** re-ranks the
top-N candidates.

## Why it beats vanilla CLIP
A CLIP bi-encoder scores image and text independently (one vector each), so context
("in a modern office") and attribute binding leak. BLIP-ITM is a cross-encoder: it feeds
image patches and query tokens through joint cross-attention and outputs `P(match)`, which
resolves context/binding a bi-encoder blurs. It is ~100× heavier per pair, so it runs
**only** on the stage-1 top-N — keeping the system scalable.

```
final_score = α · clip_cosine + (1 − α) · blip_itm_prob
```

## Layout (logic separated from data)
```
src/
  config.py      # paths, model ids, device, two-stage knobs (top_n, alpha)
  backbone.py    # FashionCLIP wrapper (stage-1 recall)
  indexer.py     # PART A: encode images -> ChromaDB + .npz cache
  reranker.py    # BLIP-ITM cross-encoder (stage-2 precision)
  retriever.py   # PART B: two-stage retrieve + fuse
  evaluate.py    # 5 assessment queries, grids, before/after comparison
fashion_retrieval_approach2.ipynb
artifacts/       # (generated) vector store + embedding cache
results/         # (generated) result grids per query
```
Data lives in `../dataset/images_1000/`.

## Run
```bash
pip install -r requirements.txt
jupyter lab fashion_retrieval_approach2.ipynb
```
Or from Python:
```python
from src.indexer import build_index; build_index()
from src.retriever import TwoStageRetriever
r = TwoStageRetriever(top_n=50, alpha=0.5)
r.search("A red tie and a white shirt in a formal setting.", k=5, rerank=True)
```

## Tuning
- `STAGE1_TOPN` — candidates to rerank (recall ceiling vs. cost).
- `RERANK_ALPHA` — blend of CLIP similarity vs. cross-encoder precision.
- `search(..., rerank=False)` gives the CLIP-only baseline for comparison.

## Scalability to 1M images
Stage 1 is sub-linear ANN over the full DB (HNSW in Chroma). Stage 2 is O(N) with N≈50,
**independent of DB size**, so per-query cost is unchanged at 1M images. Index building is
a one-time offline GPU batch job.
