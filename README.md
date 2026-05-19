# AI-Powered Visual Search & Product Discovery Engine

This project is an end-to-end AI-powered fashion product discovery system that allows users to search products using images, text, voice, or a combination of image and text queries. The application uses semantic vector search powered by FAISS and CLIP embeddings to retrieve visually and contextually similar fashion products.

The system was designed to provide a modern e-commerce style search experience where users can naturally search for fashion items such as:

- “red linen shirt under ₹1500”
- “black sneakers”
- “white oversized hoodie”

Users can also upload a fashion image and retrieve visually similar products instantly.

---

# Project Overview

The application combines:

- Computer Vision
- Natural Language Processing
- Vector Similarity Search
- FastAPI Backend
- Modern Frontend UI
- AI-based Query Filtering

to create an intelligent visual search engine for fashion products.

The project supports:

- Image Search
- Text Search
- Hybrid Image + Text Search
- Voice Search
- Semantic Product Retrieval
- Metadata Filtering

---

# How the System Works

The workflow of the system is divided into multiple stages.

## 1. User Input

The user can interact with the system in multiple ways:

- Upload a fashion image
- Enter a natural language query
- Use voice search
- Combine image and text together

Example:

- Upload shoe image + query “black running shoes”

---

## 2. Embedding Generation

The application converts user inputs into vector embeddings.

### Image Embeddings

Uploaded images are processed using CLIP-based embeddings.  
These embeddings capture the visual semantics of the fashion product.

### Text Embeddings

Text queries are converted into semantic embeddings using embedding models.

### Hybrid Embeddings

For combined image + text search:

- image embedding
- text embedding

are merged into a single semantic representation.

---

## 3. Vector Search using FAISS

All fashion products are indexed using FAISS.

FAISS performs:

- nearest-neighbor search
- cosine similarity matching
- semantic retrieval

This allows the system to retrieve visually and semantically similar products efficiently.

The project uses:

- `faiss.index`
- `image_paths.npy`

for vector indexing and image mapping.

---

## 4. Metadata Filtering

The system also applies AI-based metadata filtering.

Natural language queries are analyzed to extract:

- color
- category
- brand
- price constraints

Example:

Query:
```text
red shirt under ₹1500
