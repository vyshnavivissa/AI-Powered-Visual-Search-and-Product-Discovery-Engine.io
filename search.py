"""
search.py  —  FAISS search + metadata filtering.

Index: IndexFlatIP (inner product = cosine sim on normalised vectors).
Embeddings: CLIP 512-d (images) / OpenAI text-embedding-3-small 512-d (text).
"""

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

EXPECTED_DIM = 512

# ============================
# LOAD METADATA
# ============================
df = pd.read_csv("fashion_subset.csv")
df.columns = df.columns.str.strip()

metadata_dict: dict[str, dict] = {}
for _, row in df.iterrows():
    key = str(row["id"]) + ".jpg"
    product_name = str(row.get("productDisplayName", ""))
    derived_brand = product_name.split()[0].lower() if product_name else "unknown"
    metadata_dict[key] = {
        "price":        float(row.get("price", 0)),
        "color":        str(row.get("baseColour", "unknown")).lower().strip(),
        "category":     str(row.get("articleType", "unknown")).lower().strip(),
        "brand":        derived_brand,
        "product_name": product_name,
    }

print(f"Metadata loaded: {len(metadata_dict)} items.")

# ============================
# LOAD FAISS INDEX
# ============================
index = faiss.read_index("faiss.index")
image_paths = np.load("image_paths.npy", allow_pickle=True)

if index.d != EXPECTED_DIM:
    raise ValueError(
        f"FAISS index dim={index.d} but model outputs dim={EXPECTED_DIM}. "
        f"Delete faiss.index and run index_data.py again."
    )

print(f"FAISS index: dim={index.d}, vectors={index.ntotal}")


# ============================
# SEARCH
# ============================
def search_similar(embedding: list | np.ndarray, top_k: int = 20) -> list[str]:
    emb = np.array(embedding, dtype=np.float32).reshape(1, -1)

    if emb.shape[1] != index.d:
        raise ValueError(
            f"Query dim={emb.shape[1]} != index dim={index.d}. "
            f"Check model.py and index_data.py use the same embedding dim."
        )

    # Normalise before inner-product search (cosine similarity)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    scores, indices = index.search(emb, top_k)
    return [image_paths[i] for i in indices[0] if i != -1]


# ============================
# FILTER
# ============================
def apply_filters(results: list[str], filters: dict) -> list[dict]:
    """Filter FAISS results by metadata and return enriched dicts."""
    out = []
    for img in results:
        img = str(img)
        item = metadata_dict.get(img)
        if not item:
            continue

        if filters.get("max_price") and item["price"] > filters["max_price"]:
            continue
        if filters.get("color") and filters["color"].lower() not in item["color"]:
            continue
        if filters.get("category") and filters["category"].lower() not in item["category"]:
            continue
        if filters.get("brand"):
            bq = filters["brand"].lower()
            if bq not in item["brand"] and bq not in item["product_name"].lower():
                continue

        out.append({
            "image":        img,
            "image_url":    f"{BACKEND_URL}/images/{img}",
            "price":        item["price"],
            "color":        item["color"],
            "category":     item["category"],
            "brand":        item["brand"],
            "product_name": item["product_name"],
        })
    return out