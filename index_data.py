"""
index_data.py — rebuild FAISS index from fashion images using CLIP.
Run once: python index_data.py
"""

import os
import numpy as np
import faiss
from PIL import Image
from model import get_image_embedding

IMAGE_FOLDER = "fashion"
OUT_INDEX    = "faiss.index"
OUT_PATHS    = "image_paths.npy"

if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError(f"Folder '{IMAGE_FOLDER}' not found.")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

embeddings  = []
image_paths = []

files = sorted(os.listdir(IMAGE_FOLDER))
total = sum(1 for f in files if any(f.lower().endswith(e) for e in VALID_EXTENSIONS))
print(f"Found {total} images to index.")

for i, file in enumerate(files, 1):
    if not any(file.lower().endswith(ext) for ext in VALID_EXTENSIONS):
        continue
    path = os.path.join(IMAGE_FOLDER, file)
    try:
        image = Image.open(path).convert("RGB")
        emb = get_image_embedding(image)
        if emb is None:
            print(f"  [{i}/{total}] SKIP: {file}")
            continue
        flat = np.array(emb, dtype=np.float32).flatten()
        if flat.shape[0] != 512:
            print(f"  [{i}/{total}] SKIP wrong dim={flat.shape[0]}: {file}")
            continue
        embeddings.append(flat)
        image_paths.append(file)
        if i % 100 == 0 or i == total:
            print(f"  [{i}/{total}] indexed {file}")
    except Exception as e:
        print(f"  [{i}/{total}] ERROR {file}: {e}")

if not embeddings:
    raise ValueError("No embeddings generated.")

embeddings = np.stack(embeddings).astype(np.float32)
print(f"\nFinal embeddings shape: {embeddings.shape}")  # should be (N, 512)

index = faiss.IndexFlatIP(512)
index.add(embeddings)

faiss.write_index(index, OUT_INDEX)
np.save(OUT_PATHS, np.array(image_paths))
print(f"Done — {index.ntotal} vectors saved to '{OUT_INDEX}'.")