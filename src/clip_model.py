"""Loads and wraps a CLIP model for encoding video frames and text queries.

CLIP is a CNN/transformer vision-language model trained to place matching images and
text descriptions close together in a shared embedding space. That makes it a good fit
for open-ended queries like "a red car" or "a woman in a red dress" without needing a
separate model per object class.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import DEFAULT_MODEL_NAME


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEncoder:
    """Thin wrapper around a HuggingFace CLIP model for image/text embeddings."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.device = device or get_device()
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images_bgr: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of OpenCV BGR images into L2-normalized CLIP embeddings."""
        pil_images = [Image.fromarray(img[:, :, ::-1]) for img in images_bgr]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        # transformers >=5 returns a BaseModelOutputWithPooling; the projected embedding
        # lives in .pooler_output. Older versions returned the tensor directly.
        features = getattr(outputs, "pooler_output", outputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, query: str) -> np.ndarray:
        """Encode a natural-language query into an L2-normalized CLIP embedding."""
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        # transformers >=5 returns a BaseModelOutputWithPooling; the projected embedding
        # lives in .pooler_output. Older versions returned the tensor directly.
        features = getattr(outputs, "pooler_output", outputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)[0]


@lru_cache(maxsize=2)
def get_shared_encoder(model_name: str = DEFAULT_MODEL_NAME) -> ClipEncoder:
    """Cache encoders by model name so repeated calls (e.g. from a UI) reuse the loaded model."""
    return ClipEncoder(model_name=model_name)
