from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import io
from typing import Optional
from dotenv import load_dotenv
import os

from model import get_image_embedding, get_text_embedding
from search import search_similar, apply_filters, metadata_dict
from llm import extract_filters

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASHION_DIR = os.path.join(BASE_DIR, "fashion")

app.mount("/images", StaticFiles(directory=FASHION_DIR), name="images")


def safe_image_embedding(image) -> np.ndarray:
    emb = get_image_embedding(image)
    if emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate image embedding.")
    return np.array(emb, dtype=np.float32).flatten()


def safe_text_embedding(text: str) -> np.ndarray:
    emb = get_text_embedding(text)
    if emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate text embedding.")
    return np.array(emb, dtype=np.float32).flatten()


def build_result(img, backend_url: str) -> dict | None:
    item = metadata_dict.get(str(img))
    if not item:
        return None
    return {
        "image":        img,
        "image_url":    f"{backend_url}/images/{img}",
        "price":        item["price"],
        "color":        item["color"],
        "category":     item["category"],
        "brand":        item["brand"],
        "product_name": item["product_name"],
    }


@app.get("/")
def home():
    return {"message": "Backend is running successfully"}


@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.post("/search/")
async def search(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None)
):
    try:
        # CASE 1: IMAGE + TEXT
        if file and query:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_emb = safe_image_embedding(image)
            text_emb  = safe_text_embedding(query)
            alpha = 0.6
            combined_emb = alpha * image_emb + (1 - alpha) * text_emb
            norm = np.linalg.norm(combined_emb)
            if norm == 0:
                raise HTTPException(status_code=500, detail="Combined embedding is zero vector.")
            combined_emb = combined_emb / norm
            results = search_similar(combined_emb.tolist(), top_k=20)
            filters = extract_filters(query) if len(query.split()) > 2 else {}
            filtered_results = apply_filters(results, filters)
            if not filtered_results:
                filtered_results = apply_filters(results, {})
            return {"type": "image + text", "filters": filters, "results": filtered_results}

        # CASE 2: IMAGE ONLY
        elif file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            emb = safe_image_embedding(image)
            results = search_similar(emb.tolist(), top_k=20)
            full_results = [r for r in (build_result(img, BACKEND_URL) for img in results) if r is not None]
            return {"type": "image", "filters": {}, "results": full_results}

        # CASE 3: TEXT ONLY
        elif query:
            emb = safe_text_embedding(query)
            results = search_similar(emb.tolist(), top_k=20)
            filters = extract_filters(query) if len(query.split()) > 2 else {}
            filtered_results = apply_filters(results, filters)
            if not filtered_results:
                filtered_results = apply_filters(results, {})
            return {"type": "text", "filters": filters, "results": filtered_results}

        else:
            return {"error": "Provide an image or a query"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}