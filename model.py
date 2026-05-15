import os
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 512

_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("CLIP model loaded.")
    return _clip_model, _clip_processor


def _to_vector(raw) -> np.ndarray:
    """
    Safely convert whatever CLIP returns into a flat float32 numpy array.
    Handles: torch.Tensor, BaseModelOutputWithPooling, or numpy array.
    """
    import torch

    # If it's a model output object, pull the pooler_output tensor
    if hasattr(raw, "pooler_output"):
        raw = raw.pooler_output
    elif hasattr(raw, "last_hidden_state"):
        # fallback: use CLS token (first token)
        raw = raw.last_hidden_state[:, 0, :]

    # Now it should be a tensor
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()

    return np.array(raw, dtype=np.float32).flatten()


def get_image_embedding(image: Image.Image) -> np.ndarray | None:
    try:
        import torch
        model, processor = _load_clip()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            # Try the high-level method first; fall back to sub-model
            try:
                raw = model.get_image_features(pixel_values=inputs["pixel_values"])
            except Exception:
                raw = model.vision_model(pixel_values=inputs["pixel_values"])

        emb = _to_vector(raw)

        # If projection gave 512, good. If sub-model gave 768, project manually.
        if emb.shape[0] == 768:
            proj = model.visual_projection.weight.detach().cpu().numpy()  # (512, 768)
            emb = proj @ emb  # → (512,)

        norm = np.linalg.norm(emb)
        emb = (emb / norm) if norm > 0 else emb
        print(f"Image emb shape={emb.shape}")
        return emb
    except Exception as e:
        import traceback
        print(f"Image embedding error: {e}")
        traceback.print_exc()
        return None


def get_text_embedding(text: str) -> np.ndarray | None:
    try:
        import torch
        model, processor = _load_clip()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            try:
                raw = model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            except Exception:
                raw = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

        emb = _to_vector(raw)

        # If sub-model gave 768, project manually.
        if emb.shape[0] == 768:
            proj = model.text_projection.weight.detach().cpu().numpy()  # (512, 768)
            emb = proj @ emb  # → (512,)

        norm = np.linalg.norm(emb)
        emb = (emb / norm) if norm > 0 else emb
        print(f"Text emb shape={emb.shape}")
        return emb
    except Exception as e:
        import traceback
        print(f"Text embedding error: {e}")
        traceback.print_exc()
        return None