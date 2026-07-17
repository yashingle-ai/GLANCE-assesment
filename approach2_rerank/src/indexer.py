"""PART A — The Indexer (stage-1 store).

Encodes images with FashionCLIP and persists them to ChromaDB (used as the ANN
recall index) plus an .npz cache. Stage 2 (reranking) needs no index — it reads
the raw candidate images on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from . import config
from .backbone import get_backbone


def list_images(image_dir: Path = config.IMAGE_DIR) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)


def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index(image_dir: Path = config.IMAGE_DIR, force: bool = False) -> dict:
    paths = list_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    ids = [p.name for p in paths]

    if config.EMB_CACHE.exists() and not force:
        cached = np.load(config.EMB_CACHE, allow_pickle=True)
        if list(cached["ids"]) == ids and get_collection().count() == len(ids):
            print(f"[indexer] cache hit: {len(ids)} embeddings reused.")
            return {"ids": ids, "embeddings": cached["embeddings"]}

    print(f"[indexer] encoding {len(paths)} images with FashionCLIP ...")
    embeddings = get_backbone().encode_images(paths).astype("float32")
    np.savez(config.EMB_CACHE, ids=np.array(ids), embeddings=embeddings)

    try:
        import chromadb

        chromadb.PersistentClient(path=str(config.CHROMA_DIR)).delete_collection(
            config.COLLECTION_NAME
        )
    except Exception:
        pass
    col = get_collection()
    B = 512
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i : i + B],
            embeddings=embeddings[i : i + B].tolist(),
            metadatas=[{"filename": f} for f in ids[i : i + B]],
        )
    print(f"[indexer] done. {len(ids)} vectors in Chroma + {config.EMB_CACHE.name}.")
    return {"ids": ids, "embeddings": embeddings}
