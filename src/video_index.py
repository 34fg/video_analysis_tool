"""Build, save and load a searchable CLIP embedding index for a video."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .clip_model import ClipEncoder, get_shared_encoder
from .config import DEFAULT_BATCH_SIZE, DEFAULT_SAMPLE_FPS, DEFAULT_THUMB_MAX_SIZE
from .frame_extractor import iter_sampled_frames


@dataclass
class VideoIndex:
    video_path: str
    model_name: str
    sample_fps: float
    timestamps: np.ndarray       # shape (N,), seconds
    embeddings: np.ndarray       # shape (N, D), L2-normalized CLIP embeddings
    thumbnail_paths: List[str]   # length N, one JPEG thumbnail per sampled frame

    def save(self, index_path: str) -> None:
        parent = os.path.dirname(os.path.abspath(index_path))
        os.makedirs(parent, exist_ok=True)
        np.savez(
            index_path,
            video_path=self.video_path,
            model_name=self.model_name,
            sample_fps=self.sample_fps,
            timestamps=self.timestamps,
            embeddings=self.embeddings,
            thumbnail_paths=np.array(self.thumbnail_paths, dtype=object),
        )

    @staticmethod
    def load(index_path: str) -> "VideoIndex":
        data = np.load(index_path, allow_pickle=True)
        return VideoIndex(
            video_path=str(data["video_path"]),
            model_name=str(data["model_name"]),
            sample_fps=float(data["sample_fps"]),
            timestamps=data["timestamps"],
            embeddings=data["embeddings"],
            thumbnail_paths=list(data["thumbnail_paths"]),
        )


def _save_thumbnail(image_bgr: np.ndarray, path: str, max_size: int) -> None:
    h, w = image_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)


def build_video_index(
    video_path: str,
    thumbnails_dir: str,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    model_name: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thumb_max_size: int = DEFAULT_THUMB_MAX_SIZE,
    encoder: Optional[ClipEncoder] = None,
    progress: bool = True,
) -> VideoIndex:
    """Extract frames from `video_path`, embed them with CLIP, and return a VideoIndex."""
    if encoder is None:
        encoder = get_shared_encoder(model_name) if model_name else get_shared_encoder()

    timestamps: List[float] = []
    thumbnail_paths: List[str] = []
    embeddings_batches: List[np.ndarray] = []
    batch_images: List[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        embeddings_batches.append(encoder.encode_images(batch_images))
        batch_images.clear()

    frame_iter = iter_sampled_frames(video_path, sample_fps=sample_fps)
    if progress:
        frame_iter = tqdm(frame_iter, desc="Extracting & embedding frames", unit="frame")

    for sampled in frame_iter:
        thumb_path = os.path.join(thumbnails_dir, f"frame_{sampled.index:06d}.jpg")
        _save_thumbnail(sampled.image_bgr, thumb_path, thumb_max_size)

        timestamps.append(sampled.timestamp)
        thumbnail_paths.append(thumb_path)
        batch_images.append(sampled.image_bgr)

        if len(batch_images) >= batch_size:
            flush_batch()
    flush_batch()

    if not embeddings_batches:
        raise ValueError(f"No frames could be extracted from {video_path}")

    embeddings = np.concatenate(embeddings_batches, axis=0)
    return VideoIndex(
        video_path=video_path,
        model_name=encoder.model_name,
        sample_fps=sample_fps,
        timestamps=np.array(timestamps, dtype=np.float32),
        embeddings=embeddings,
        thumbnail_paths=thumbnail_paths,
    )
